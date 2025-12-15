"""Download and process NCCPI soil productivity data from NRCS Soil Data Access."""

from pathlib import Path
import time

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import Progress, TaskID

from ag_rents.config import get_settings

console = Console()

# NRCS Soil Data Access URL
SOIL_DATA_ACCESS_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"


def query_soil_data_access(sql_query: str) -> pd.DataFrame:
    """Execute SQL query against NRCS Soil Data Access service.

    Args:
        sql_query: SQL query to execute against SDA tables

    Returns:
        DataFrame with query results
    """
    payload = {
        "format": "JSON",
        "query": sql_query.strip()
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(SOIL_DATA_ACCESS_URL, json=payload)
        response.raise_for_status()
        result = response.json()

    if "Table" not in result or not result["Table"]:
        return pd.DataFrame()

    # Parse JSON response into DataFrame
    data = result["Table"]
    if not data:
        return pd.DataFrame()

    # First row contains column names
    columns = list(data[0].keys())
    df = pd.DataFrame(data, columns=columns)

    return df


def download_nccpi_by_county(
    state_fips: str,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Download NCCPI values for all counties in a state.

    The NCCPI (National Commodity Crop Productivity Index) measures inherent
    soil productivity for various crops on a 0-1 scale.

    Args:
        state_fips: 2-digit state FIPS code
        output_dir: Directory to save raw data

    Returns:
        DataFrame with county-level NCCPI values
    """
    # Query county-level area-weighted NCCPI values
    # This query joins mapunit, component, and cointerp tables to get NCCPI
    sql_query = f"""
    SELECT
        legend.areasymbol AS areasymbol,
        legend.areaname AS county_name,
        SUM(mapunit.muacres * component.comppct_r / 100 *
            CASE WHEN cointerp.interprating IS NULL THEN 0
                 ELSE cointerp.interprating END) /
            NULLIF(SUM(mapunit.muacres * component.comppct_r / 100), 0) AS nccpi_weighted_avg,
        SUM(mapunit.muacres) AS total_acres
    FROM
        legend
        INNER JOIN mapunit ON legend.lkey = mapunit.lkey
        INNER JOIN component ON mapunit.mukey = component.mukey
        LEFT JOIN cointerp ON component.cokey = cointerp.cokey
            AND cointerp.mrulename = 'NCCPI - National Commodity Crop Productivity Index (Ver 3.0)'
    WHERE
        legend.areasymbol LIKE '{state_fips}%'
        AND mapunit.muacres > 0
    GROUP BY
        legend.areasymbol, legend.areaname
    ORDER BY
        legend.areasymbol
    """

    df = query_soil_data_access(sql_query)

    if df.empty:
        console.print(f"[yellow]No NCCPI data for state {state_fips}[/yellow]")
        return df

    # Parse area symbol to extract county FIPS
    # Format is typically "STATE###" where ### is county code
    df["state_fips"] = state_fips
    df["county_fips_suffix"] = df["areasymbol"].str[-3:]
    df["county_fips"] = df["state_fips"] + df["county_fips_suffix"]

    # Convert numeric columns
    for col in ["nccpi_weighted_avg", "total_acres"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_all_nccpi(output_dir: Path | None = None) -> pd.DataFrame:
    """Download NCCPI for all CONUS states.

    Args:
        output_dir: Directory to save raw data

    Returns:
        Combined DataFrame with all county NCCPI values
    """
    settings = get_settings()
    output_dir = output_dir or settings.raw_dir / "gssurgo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CONUS state FIPS codes
    state_fips_codes = [
        "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
        "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
        "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
        "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
        "47", "48", "49", "50", "51", "53", "54", "55", "56"
    ]

    all_data = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Downloading NCCPI...", total=len(state_fips_codes))

        for state_fips in state_fips_codes:
            try:
                df = download_nccpi_by_county(state_fips)
                if not df.empty:
                    all_data.append(df)
                    console.print(f"[green]State {state_fips}: {len(df)} counties[/green]")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                console.print(f"[red]Error for state {state_fips}: {e}[/red]")

            progress.update(task, advance=1)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_file = output_dir / "nccpi_by_county.csv"
        combined.to_csv(output_file, index=False)
        console.print(f"[bold green]Saved {len(combined)} county NCCPI records to {output_file}[/bold green]")
        return combined

    return pd.DataFrame()


def download_nccpi_corn_soy(output_dir: Path | None = None) -> pd.DataFrame:
    """Download crop-specific NCCPI for corn and soybeans.

    This provides separate productivity indices for major crops,
    which is useful for more accurate rent predictions.

    Args:
        output_dir: Directory to save raw data

    Returns:
        DataFrame with crop-specific NCCPI values
    """
    settings = get_settings()
    output_dir = output_dir or settings.raw_dir / "gssurgo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Query for crop-specific NCCPI
    sql_query = """
    SELECT
        legend.areasymbol,
        legend.areaname AS county_name,
        cointerp.mrulename AS nccpi_type,
        SUM(mapunit.muacres * component.comppct_r / 100 *
            CASE WHEN cointerp.interprating IS NULL THEN 0
                 ELSE cointerp.interprating END) /
            NULLIF(SUM(mapunit.muacres * component.comppct_r / 100), 0) AS nccpi_weighted_avg,
        SUM(mapunit.muacres) AS total_acres
    FROM
        legend
        INNER JOIN mapunit ON legend.lkey = mapunit.lkey
        INNER JOIN component ON mapunit.mukey = component.mukey
        LEFT JOIN cointerp ON component.cokey = cointerp.cokey
    WHERE
        legend.areasymbol LIKE 'IA%'
        AND mapunit.muacres > 0
        AND cointerp.mrulename IN (
            'NCCPI - NCCPI Corn and Soybeans Submodel (I)',
            'NCCPI - NCCPI Small Grains Submodel (II)',
            'NCCPI - NCCPI Cotton Submodel (III)'
        )
    GROUP BY
        legend.areasymbol, legend.areaname, cointerp.mrulename
    ORDER BY
        legend.areasymbol, cointerp.mrulename
    """

    # This is a sample query for Iowa; full implementation would loop through states
    df = query_soil_data_access(sql_query)
    return df


def load_or_download_nccpi(output_dir: Path | None = None) -> pd.DataFrame:
    """Load NCCPI data, downloading if necessary.

    Args:
        output_dir: Directory for data files

    Returns:
        DataFrame with county-level NCCPI values
    """
    settings = get_settings()
    output_dir = output_dir or settings.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_file = output_dir / "nccpi_county.parquet"

    # Check for processed file
    if processed_file.exists():
        console.print(f"[blue]Loading processed NCCPI data from {processed_file}[/blue]")
        return pd.read_parquet(processed_file)

    # Download and process
    console.print("[blue]Downloading NCCPI data from NRCS Soil Data Access...[/blue]")
    console.print("[yellow]Note: This may take several minutes due to API rate limits.[/yellow]")

    df = download_all_nccpi()

    if df.empty:
        console.print("[red]Failed to download NCCPI data[/red]")
        return df

    # Standardize columns
    df_clean = df[["county_fips", "state_fips", "county_name",
                   "nccpi_weighted_avg", "total_acres"]].copy()
    df_clean = df_clean.rename(columns={"nccpi_weighted_avg": "nccpi"})

    # Save processed file
    df_clean.to_parquet(processed_file, index=False)
    console.print(f"[green]Saved processed NCCPI to {processed_file}[/green]")

    return df_clean
