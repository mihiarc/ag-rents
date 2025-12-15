"""Download and process Census TIGER/Line county boundaries."""

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import geopandas as gpd
import httpx
from rich.console import Console
from rich.progress import Progress

from ag_rents.config import get_settings

console = Console()

# Census TIGER/Line county boundaries URL (2023 vintage)
TIGER_COUNTY_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"

# CONUS state FIPS codes (excludes AK=02, HI=15, and territories)
CONUS_STATE_FIPS = [
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56"
]


def download_county_boundaries(output_dir: Path | None = None) -> gpd.GeoDataFrame:
    """Download Census TIGER/Line county boundaries.

    Args:
        output_dir: Directory to save downloaded shapefile

    Returns:
        GeoDataFrame with county boundaries
    """
    settings = get_settings()
    output_dir = output_dir or settings.raw_dir / "tiger"
    output_dir.mkdir(parents=True, exist_ok=True)

    shapefile_path = output_dir / "tl_2023_us_county.shp"

    # Check if already downloaded
    if shapefile_path.exists():
        console.print(f"[blue]Loading existing county boundaries from {shapefile_path}[/blue]")
        return gpd.read_file(shapefile_path)

    console.print("[blue]Downloading Census TIGER/Line county boundaries...[/blue]")

    with Progress() as progress:
        task = progress.add_task("[cyan]Downloading...", total=None)

        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(TIGER_COUNTY_URL)
            response.raise_for_status()

        progress.update(task, completed=100)

    # Extract shapefile from zip
    console.print("[blue]Extracting shapefile...[/blue]")
    with ZipFile(BytesIO(response.content)) as zf:
        zf.extractall(output_dir)

    # Load and return
    gdf = gpd.read_file(shapefile_path)
    console.print(f"[green]Loaded {len(gdf)} county boundaries[/green]")

    return gdf


def filter_conus_counties(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter to contiguous US counties only.

    Args:
        gdf: GeoDataFrame with all US counties

    Returns:
        GeoDataFrame with only CONUS counties
    """
    # STATEFP is the state FIPS code column in TIGER data
    gdf_conus = gdf[gdf["STATEFP"].isin(CONUS_STATE_FIPS)].copy()
    console.print(f"[blue]Filtered to {len(gdf_conus)} CONUS counties[/blue]")
    return gdf_conus


def standardize_county_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardize county boundary columns.

    Args:
        gdf: Raw TIGER/Line GeoDataFrame

    Returns:
        Standardized GeoDataFrame with consistent column names
    """
    # Select and rename relevant columns
    columns_map = {
        "STATEFP": "state_fips",
        "COUNTYFP": "county_fips_suffix",
        "GEOID": "county_fips",
        "NAME": "county_name",
        "NAMELSAD": "county_name_full",
        "ALAND": "land_area_sqm",
        "AWATER": "water_area_sqm",
        "geometry": "geometry",
    }

    gdf_std = gdf[list(columns_map.keys())].copy()
    gdf_std = gdf_std.rename(columns=columns_map)

    # Convert areas to square miles
    gdf_std["land_area_sqmi"] = gdf_std["land_area_sqm"] / 2_589_988.11
    gdf_std["water_area_sqmi"] = gdf_std["water_area_sqm"] / 2_589_988.11

    # Ensure CRS is WGS84
    if gdf_std.crs != "EPSG:4326":
        gdf_std = gdf_std.to_crs("EPSG:4326")

    return gdf_std


def get_county_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add centroid coordinates to county boundaries.

    Args:
        gdf: County boundaries GeoDataFrame

    Returns:
        GeoDataFrame with centroid lat/lon columns added
    """
    gdf = gdf.copy()

    # Project to Albers Equal Area for accurate centroid calculation
    gdf_projected = gdf.to_crs("ESRI:102003")  # USA Contiguous Albers Equal Area
    centroids_projected = gdf_projected.geometry.centroid

    # Transform centroids back to WGS84 for lat/lon
    centroids_wgs84 = gpd.GeoSeries(centroids_projected, crs="ESRI:102003").to_crs("EPSG:4326")
    gdf["centroid_lon"] = centroids_wgs84.x
    gdf["centroid_lat"] = centroids_wgs84.y

    return gdf


def load_or_download_counties(output_dir: Path | None = None) -> gpd.GeoDataFrame:
    """Load county boundaries, downloading if necessary.

    This is the main entry point for getting county boundaries.

    Args:
        output_dir: Directory for data files

    Returns:
        Processed GeoDataFrame with CONUS county boundaries
    """
    settings = get_settings()
    output_dir = output_dir or settings.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_file = output_dir / "county_boundaries.parquet"

    # Check for processed file
    if processed_file.exists():
        console.print(f"[blue]Loading processed county boundaries from {processed_file}[/blue]")
        return gpd.read_parquet(processed_file)

    # Download and process
    gdf = download_county_boundaries()
    gdf = filter_conus_counties(gdf)
    gdf = standardize_county_boundaries(gdf)
    gdf = get_county_centroids(gdf)

    # Save processed file
    gdf.to_parquet(processed_file)
    console.print(f"[green]Saved processed boundaries to {processed_file}[/green]")

    return gdf
