#!/usr/bin/env python3
"""Create visualizations for agricultural land rents data.

This script generates:
1. Choropleth maps of cash rents by county
2. Time series plots of rent trends
3. Distribution histograms
4. Interactive Folium maps

Usage:
    uv run python scripts/create_visualizations.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

from ag_rents.config import get_settings

console = Console()

# Color schemes for maps
CROPLAND_CMAP = "YlGn"
PASTURE_CMAP = "YlOrBr"


def create_static_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Create a static choropleth map.

    Args:
        gdf: GeoDataFrame with county boundaries and data
        column: Column to visualize
        title: Map title
        output_path: Path to save figure
        cmap: Matplotlib colormap
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plot counties without data in gray
    gdf[gdf[column].isna()].plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.1)

    # Plot counties with data
    gdf[gdf[column].notna()].plot(
        ax=ax,
        column=column,
        cmap=cmap,
        legend=True,
        legend_kwds={"label": "$/acre", "orientation": "horizontal", "shrink": 0.6},
        edgecolor="white",
        linewidth=0.1,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved map to {output_path}[/green]")


def create_time_series_plot(
    df: pd.DataFrame,
    output_path: Path,
):
    """Create time series plot of average cash rents.

    Args:
        df: Panel DataFrame with year column
        output_path: Path to save figure
    """
    if "year" not in df.columns:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Calculate annual means
    rent_cols = ["cropland_rent_nonirrigated", "cropland_rent_irrigated", "pasture_rent"]
    rent_cols = [c for c in rent_cols if c in df.columns]

    for col in rent_cols:
        annual_mean = df.groupby("year")[col].mean()
        label = col.replace("_", " ").replace("cropland rent ", "Cropland ").replace("pasture rent", "Pasture")
        ax.plot(annual_mean.index, annual_mean.values, marker="o", label=label, linewidth=2)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average Cash Rent ($/acre)", fontsize=12)
    ax.set_title("Average Cash Rents Over Time", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    console.print(f"[green]Saved time series plot to {output_path}[/green]")


def create_distribution_plot(
    df: pd.DataFrame,
    output_path: Path,
):
    """Create histogram of cash rent distributions.

    Args:
        df: DataFrame with rent columns
        output_path: Path to save figure
    """
    rent_cols = ["cropland_rent_nonirrigated", "cropland_rent_irrigated", "pasture_rent"]
    rent_cols = [c for c in rent_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(rent_cols), figsize=(5 * len(rent_cols), 5))
    if len(rent_cols) == 1:
        axes = [axes]

    colors = ["#2E8B57", "#4169E1", "#CD853F"]

    for ax, col, color in zip(axes, rent_cols, colors):
        data = df[col].dropna()
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: ${data.mean():.0f}")
        ax.axvline(data.median(), color="black", linestyle="-", label=f"Median: ${data.median():.0f}")

        title = col.replace("_", " ").replace("cropland rent ", "Cropland ").replace("pasture rent", "Pasture")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Cash Rent ($/acre)")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    console.print(f"[green]Saved distribution plot to {output_path}[/green]")


def create_folium_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
):
    """Create interactive Folium map.

    Args:
        gdf: GeoDataFrame with county boundaries and data
        column: Column to visualize
        title: Map title
        output_path: Path to save HTML file
    """
    try:
        import folium
        from folium.plugins import Fullscreen
    except ImportError:
        console.print("[yellow]Folium not available for interactive maps[/yellow]")
        return

    # Filter to counties with data
    gdf_data = gdf[gdf[column].notna()].copy()

    if gdf_data.empty:
        console.print(f"[yellow]No data for {column}, skipping interactive map[/yellow]")
        return

    # Calculate center
    bounds = gdf_data.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # Create map
    m = folium.Map(location=center, zoom_start=4, tiles="cartodbpositron")

    # Add choropleth
    folium.Choropleth(
        geo_data=gdf_data.__geo_interface__,
        name=title,
        data=gdf_data,
        columns=["county_fips", column],
        key_on="feature.properties.county_fips",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{title} ($/acre)",
    ).add_to(m)

    # Add tooltips
    folium.GeoJson(
        gdf_data,
        style_function=lambda x: {"fillColor": "transparent", "color": "transparent"},
        tooltip=folium.GeoJsonTooltip(
            fields=["county_name", "state_fips", column],
            aliases=["County", "State FIPS", "Rent ($/acre)"],
            localize=True,
        ),
    ).add_to(m)

    Fullscreen().add_to(m)
    folium.LayerControl().add_to(m)

    m.save(str(output_path))
    console.print(f"[green]Saved interactive map to {output_path}[/green]")


def main():
    """Main visualization pipeline."""
    console.print(Panel(
        "[bold green]Agricultural Land Rents Visualization[/bold green]",
        title="ag-rents"
    ))

    settings = get_settings()
    figures_dir = settings.project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    console.print("\n[bold]Loading data...[/bold]")

    counties_file = settings.processed_dir / "county_boundaries.parquet"
    if not counties_file.exists():
        console.print("[red]County boundaries not found. Run download_data.py first.[/red]")
        return

    gdf_counties = gpd.read_parquet(counties_file)

    # Try to load output data first, fall back to processed
    output_file = settings.output_dir / "county_cash_rents_panel.parquet"
    processed_file = settings.processed_dir / "cash_rents_wide.parquet"

    if output_file.exists():
        df_rents = pd.read_parquet(output_file)
        console.print(f"[green]Loaded output data: {len(df_rents)} records[/green]")
    elif processed_file.exists():
        df_rents = pd.read_parquet(processed_file)
        console.print(f"[green]Loaded processed data: {len(df_rents)} records[/green]")
    else:
        console.print("[red]No rent data found. Run download_data.py first.[/red]")
        return

    # Merge with county boundaries for mapping
    console.print("\n[bold]Creating visualizations...[/bold]")

    # For maps, use most recent year or multi-year average
    if "year" in df_rents.columns:
        latest_year = df_rents["year"].max()
        df_map = df_rents[df_rents["year"] == latest_year].copy()
        year_label = str(latest_year)
    else:
        df_map = df_rents.copy()
        year_label = "Average"

    gdf_map = gdf_counties.merge(df_map, on="county_fips", how="left")

    # Create static maps
    console.print("\n[cyan]Creating static maps...[/cyan]")

    if "cropland_rent_nonirrigated" in gdf_map.columns:
        create_static_map(
            gdf_map,
            "cropland_rent_nonirrigated",
            f"Non-Irrigated Cropland Cash Rents ({year_label})",
            figures_dir / f"map_cropland_nonirrigated_{year_label}.png",
            cmap=CROPLAND_CMAP,
        )

    if "cropland_rent_irrigated" in gdf_map.columns:
        create_static_map(
            gdf_map,
            "cropland_rent_irrigated",
            f"Irrigated Cropland Cash Rents ({year_label})",
            figures_dir / f"map_cropland_irrigated_{year_label}.png",
            cmap=CROPLAND_CMAP,
        )

    if "pasture_rent" in gdf_map.columns:
        create_static_map(
            gdf_map,
            "pasture_rent",
            f"Pastureland Cash Rents ({year_label})",
            figures_dir / f"map_pasture_{year_label}.png",
            cmap=PASTURE_CMAP,
        )

    # Create time series plot
    if "year" in df_rents.columns:
        console.print("\n[cyan]Creating time series plot...[/cyan]")
        create_time_series_plot(df_rents, figures_dir / "timeseries_cash_rents.png")

    # Create distribution plots
    console.print("\n[cyan]Creating distribution plots...[/cyan]")
    create_distribution_plot(df_rents, figures_dir / "distributions_cash_rents.png")

    # Create interactive maps
    console.print("\n[cyan]Creating interactive maps...[/cyan]")

    if "cropland_rent_nonirrigated" in gdf_map.columns:
        create_folium_map(
            gdf_map,
            "cropland_rent_nonirrigated",
            f"Non-Irrigated Cropland Cash Rents ({year_label})",
            figures_dir / f"interactive_cropland_{year_label}.html",
        )

    console.print(Panel("[bold green]Visualization Complete[/bold green]"))
    console.print(f"\n[bold]Figures saved to {figures_dir}:[/bold]")
    for f in sorted(figures_dir.iterdir()):
        if f.is_file():
            console.print(f"  {f.name}")


if __name__ == "__main__":
    main()
