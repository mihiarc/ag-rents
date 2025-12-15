#!/usr/bin/env python3
"""Process downloaded data and create final output dataset.

This script:
1. Loads processed NASS cash rents data
2. Loads county boundaries and NCCPI data
3. Interpolates missing values
4. Creates final output dataset

Usage:
    uv run python scripts/process_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ag_rents.config import get_settings
from ag_rents.interpolation import create_output_dataset, interpolate_cash_rents

console = Console()


def load_processed_data():
    """Load all processed data files."""
    settings = get_settings()

    # Load cash rents
    rents_file = settings.processed_dir / "cash_rents_wide.parquet"
    if not rents_file.exists():
        console.print(f"[red]Cash rents data not found at {rents_file}[/red]")
        console.print("[yellow]Run download_data.py first with your NASS API key[/yellow]")
        return None, None, None

    df_rents = pd.read_parquet(rents_file)
    console.print(f"[green]Loaded {len(df_rents)} cash rent records[/green]")

    # Load county boundaries
    counties_file = settings.processed_dir / "county_boundaries.parquet"
    if not counties_file.exists():
        console.print(f"[red]County boundaries not found at {counties_file}[/red]")
        console.print("[yellow]Run download_data.py first[/yellow]")
        return df_rents, None, None

    gdf_counties = gpd.read_parquet(counties_file)
    console.print(f"[green]Loaded {len(gdf_counties)} county boundaries[/green]")

    # Load NCCPI (optional)
    nccpi_file = settings.processed_dir / "nccpi_county.parquet"
    df_nccpi = None
    if nccpi_file.exists():
        df_nccpi = pd.read_parquet(nccpi_file)
        console.print(f"[green]Loaded {len(df_nccpi)} NCCPI records[/green]")
    else:
        console.print("[yellow]NCCPI data not found - will use location-only interpolation[/yellow]")

    return df_rents, gdf_counties, df_nccpi


def summarize_data(df_rents: pd.DataFrame, gdf_counties: gpd.GeoDataFrame):
    """Print summary statistics for the data."""
    console.print(Panel("[bold cyan]Data Summary[/bold cyan]"))

    # Cash rents summary
    table = Table(title="Cash Rents by Type")
    table.add_column("Rent Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Mean ($/acre)", justify="right")
    table.add_column("Median ($/acre)", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for col in ["cropland_rent_nonirrigated", "cropland_rent_irrigated", "pasture_rent"]:
        if col in df_rents.columns:
            data = df_rents[col].dropna()
            table.add_row(
                col.replace("_", " ").title(),
                str(len(data)),
                f"${data.mean():.2f}",
                f"${data.median():.2f}",
                f"${data.min():.2f}",
                f"${data.max():.2f}",
            )

    console.print(table)

    # Coverage summary
    console.print(f"\n[bold]Geographic Coverage:[/bold]")
    console.print(f"  Total CONUS counties: {len(gdf_counties)}")
    console.print(f"  Counties with rent data: {df_rents['county_fips'].nunique()}")

    if "year" in df_rents.columns:
        console.print(f"  Years: {df_rents['year'].min()} - {df_rents['year'].max()}")


def create_yearly_summaries(df: pd.DataFrame, output_dir: Path):
    """Create year-by-year summary files."""
    if "year" not in df.columns:
        return

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year].copy()
        year_file = output_dir / f"cash_rents_{year}.csv"
        year_df.to_csv(year_file, index=False)

    console.print(f"[green]Created yearly summary files in {output_dir}[/green]")


def create_multiyear_averages(df: pd.DataFrame, output_dir: Path):
    """Create multi-year average dataset."""
    if "year" not in df.columns:
        return df

    # Calculate mean across years for each county
    rent_cols = [c for c in ["cropland_rent_nonirrigated", "cropland_rent_irrigated",
                             "pasture_rent"] if c in df.columns]

    groupby_cols = ["county_fips", "state_fips", "county_name"]
    groupby_cols = [c for c in groupby_cols if c in df.columns]

    df_avg = df.groupby(groupby_cols)[rent_cols].mean().reset_index()
    df_avg["year_range"] = f"{df['year'].min()}-{df['year'].max()}"

    avg_file = output_dir / "cash_rents_multiyear_avg.csv"
    df_avg.to_csv(avg_file, index=False)
    console.print(f"[green]Created multi-year average file: {avg_file}[/green]")

    return df_avg


def main():
    """Main processing pipeline."""
    console.print(Panel(
        "[bold green]Agricultural Land Rents Data Processing[/bold green]\n"
        "Creating final output dataset with interpolated values",
        title="ag-rents"
    ))

    settings = get_settings()

    # Load data
    console.print("\n[bold]Loading processed data...[/bold]")
    df_rents, gdf_counties, df_nccpi = load_processed_data()

    if df_rents is None or gdf_counties is None:
        console.print("[red]Cannot proceed without required data files[/red]")
        return

    # Summarize input data
    summarize_data(df_rents, gdf_counties)

    # Create output directory
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each year separately to create panel dataset
    console.print("\n[bold]Processing annual data...[/bold]")

    if "year" in df_rents.columns:
        all_years = []
        for year in sorted(df_rents["year"].unique()):
            console.print(f"[blue]Processing {year}...[/blue]")
            df_year = df_rents[df_rents["year"] == year].copy()

            # Interpolate missing values for this year
            df_interpolated = interpolate_cash_rents(
                df_year,
                gdf_counties,
                df_nccpi,
                rent_cols=["cropland_rent_nonirrigated", "cropland_rent_irrigated", "pasture_rent"],
                method="hybrid"
            )
            df_interpolated["year"] = year
            all_years.append(df_interpolated)

        # Combine all years
        df_final = pd.concat(all_years, ignore_index=True)
    else:
        # Single cross-section
        df_final = interpolate_cash_rents(
            df_rents,
            gdf_counties,
            df_nccpi,
            method="hybrid"
        )

    # Save final outputs
    console.print("\n[bold]Saving outputs...[/bold]")

    # Full panel dataset
    df_final.to_csv(output_dir / "county_cash_rents_panel.csv", index=False)
    df_final.to_parquet(output_dir / "county_cash_rents_panel.parquet", index=False)
    console.print(f"[green]Saved panel dataset to {output_dir}/county_cash_rents_panel.*[/green]")

    # Yearly files
    create_yearly_summaries(df_final, output_dir)

    # Multi-year averages
    create_multiyear_averages(df_final, output_dir)

    # Final summary
    console.print(Panel("[bold green]Processing Complete[/bold green]"))
    console.print(f"\n[bold]Output files created in {output_dir}:[/bold]")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            console.print(f"  {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
