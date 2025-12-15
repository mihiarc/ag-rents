#!/usr/bin/env python3
"""Download all data sources for agricultural land rents analysis.

This script downloads:
1. NASS Cash Rents Survey data (2008-2024)
2. Census TIGER/Line county boundaries
3. (Future) NCCPI soil productivity data

Usage:
    uv run python scripts/download_data.py [--api-key KEY]
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

from ag_rents.config import get_settings
from ag_rents.nass_client import NASSClient, clean_cash_rents, pivot_cash_rents
from ag_rents.boundaries import load_or_download_counties
from ag_rents.soil_productivity import load_or_download_nccpi

console = Console()


def download_nass_cash_rents(api_key: str | None = None) -> None:
    """Download and process NASS cash rents data."""
    console.print(Panel("[bold cyan]Downloading NASS Cash Rents Data[/bold cyan]"))

    settings = get_settings()
    client = NASSClient(api_key=api_key)

    # Download all rent types
    raw_df = client.download_all_cash_rents(
        start_year=settings.start_year,
        end_year=settings.end_year,
    )

    if raw_df.empty:
        console.print("[red]No data downloaded. Check your API key.[/red]")
        return

    # Clean the data
    console.print("\n[bold]Cleaning cash rents data...[/bold]")
    clean_df = clean_cash_rents(raw_df)

    # Save cleaned long-format data
    processed_dir = settings.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    clean_file = processed_dir / "cash_rents_clean.parquet"
    clean_df.to_parquet(clean_file, index=False)
    console.print(f"[green]Saved cleaned data to {clean_file}[/green]")

    # Pivot to wide format
    console.print("\n[bold]Pivoting to wide format...[/bold]")
    wide_df = pivot_cash_rents(clean_df)

    wide_file = processed_dir / "cash_rents_wide.parquet"
    wide_df.to_parquet(wide_file, index=False)
    console.print(f"[green]Saved wide-format data to {wide_file}[/green]")

    # Summary statistics
    console.print("\n[bold]Data Summary:[/bold]")
    console.print(f"  Years: {clean_df['year'].min()} - {clean_df['year'].max()}")
    console.print(f"  Counties with data: {clean_df['county_fips'].nunique()}")
    console.print(f"  Total observations: {len(clean_df)}")

    if "cropland_rent_nonirrigated" in wide_df.columns:
        console.print(f"\n  Cropland (non-irrigated):")
        console.print(f"    Mean: ${wide_df['cropland_rent_nonirrigated'].mean():.2f}/acre")
        console.print(f"    Median: ${wide_df['cropland_rent_nonirrigated'].median():.2f}/acre")

    if "pasture_rent" in wide_df.columns:
        console.print(f"\n  Pastureland:")
        console.print(f"    Mean: ${wide_df['pasture_rent'].mean():.2f}/acre")
        console.print(f"    Median: ${wide_df['pasture_rent'].median():.2f}/acre")


def download_county_boundaries() -> None:
    """Download Census county boundaries."""
    console.print(Panel("[bold cyan]Downloading County Boundaries[/bold cyan]"))

    gdf = load_or_download_counties()

    console.print(f"\n[bold]County Boundaries Summary:[/bold]")
    console.print(f"  Total CONUS counties: {len(gdf)}")
    console.print(f"  States: {gdf['state_fips'].nunique()}")


def download_nccpi_data() -> None:
    """Download NCCPI soil productivity data."""
    console.print(Panel("[bold cyan]Downloading NCCPI Soil Productivity Data[/bold cyan]"))
    console.print("[yellow]Note: This downloads from NRCS Soil Data Access API[/yellow]")
    console.print("[yellow]This may take 15-30 minutes due to API rate limits.[/yellow]")

    df = load_or_download_nccpi()

    if not df.empty:
        console.print(f"\n[bold]NCCPI Summary:[/bold]")
        console.print(f"  Counties with data: {len(df)}")
        console.print(f"  Mean NCCPI: {df['nccpi'].mean():.3f}")
        console.print(f"  Min NCCPI: {df['nccpi'].min():.3f}")
        console.print(f"  Max NCCPI: {df['nccpi'].max():.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download data for agricultural land rents analysis"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="NASS QuickStats API key (or set NASS_API_KEY env var)",
    )
    parser.add_argument(
        "--skip-nass",
        action="store_true",
        help="Skip NASS data download",
    )
    parser.add_argument(
        "--skip-boundaries",
        action="store_true",
        help="Skip county boundaries download",
    )
    parser.add_argument(
        "--skip-nccpi",
        action="store_true",
        help="Skip NCCPI soil productivity download",
    )
    parser.add_argument(
        "--nccpi-only",
        action="store_true",
        help="Only download NCCPI data",
    )

    args = parser.parse_args()

    console.print(Panel(
        "[bold green]Agricultural Land Rents Data Download[/bold green]\n"
        "Updating Lubowski/Mihiar methodology with NASS Cash Rents Survey",
        title="ag-rents"
    ))

    if args.nccpi_only:
        download_nccpi_data()
        console.print("\n[bold green]Download complete![/bold green]")
        return

    # Download county boundaries first (no API key needed)
    if not args.skip_boundaries:
        download_county_boundaries()
        console.print()

    # Download NASS cash rents
    if not args.skip_nass:
        download_nass_cash_rents(api_key=args.api_key)
        console.print()

    # Download NCCPI soil productivity
    if not args.skip_nccpi:
        download_nccpi_data()

    console.print("\n[bold green]Download complete![/bold green]")


if __name__ == "__main__":
    main()
