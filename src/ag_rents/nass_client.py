"""NASS QuickStats API client for downloading cash rents data."""

import time
from typing import Any
from pathlib import Path

import httpx
import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, TaskID

from ag_rents.config import get_settings

console = Console()


class NASSQueryParams(BaseModel):
    """Parameters for NASS QuickStats API query."""

    source_desc: str = "SURVEY"
    sector_desc: str = "ECONOMICS"
    group_desc: str = "EXPENSES"
    commodity_desc: str = "RENT"
    statisticcat_desc: str = "RENT"
    unit_desc: str = "$ / ACRE"
    agg_level_desc: str = "COUNTY"
    year: int | None = None
    state_alpha: str | None = None
    short_desc: str | None = None
    format: str = "JSON"


class CashRentRecord(BaseModel):
    """Validated cash rent record from NASS API."""

    state_fips_code: str = Field(alias="state_fips_code")
    county_code: str = Field(alias="county_code")
    state_alpha: str = Field(alias="state_alpha")
    state_name: str = Field(alias="state_name")
    county_name: str = Field(alias="county_name")
    year: int = Field(alias="year")
    short_desc: str = Field(alias="short_desc")
    value: str = Field(alias="Value")
    cv_percent: str | None = Field(default=None, alias="CV (%)")


class NASSClient:
    """Client for interacting with NASS QuickStats API."""

    # Short descriptions for the three rent types we need
    RENT_TYPES = {
        "cropland_nonirrigated": "RENT, CASH, CROPLAND, NON-IRRIGATED - EXPENSE, MEASURED IN $ / ACRE",
        "cropland_irrigated": "RENT, CASH, CROPLAND, IRRIGATED - EXPENSE, MEASURED IN $ / ACRE",
        "pastureland": "RENT, CASH, PASTURELAND - EXPENSE, MEASURED IN $ / ACRE",
    }

    def __init__(self, api_key: str | None = None):
        """Initialize NASS client with API key."""
        settings = get_settings()
        self.api_key = api_key or settings.nass_api_key
        self.base_url = settings.nass_base_url
        self.rate_limit_delay = settings.nass_rate_limit_delay

        if not self.api_key:
            console.print(
                "[yellow]Warning: NASS API key not set. "
                "Get one at https://quickstats.nass.usda.gov/api[/yellow]"
            )

    def _make_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make request to NASS API with rate limiting."""
        params["key"] = self.api_key
        params["format"] = "JSON"

        with httpx.Client(timeout=60.0) as client:
            response = client.get(f"{self.base_url}/api_GET/", params=params)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)  # Rate limiting
            return response.json()

    def query_cash_rents(
        self,
        rent_type: str,
        year: int | None = None,
        state: str | None = None,
    ) -> pd.DataFrame:
        """Query cash rents data for a specific rent type.

        Args:
            rent_type: One of 'cropland_nonirrigated', 'cropland_irrigated', 'pastureland'
            year: Optional year filter
            state: Optional state abbreviation filter

        Returns:
            DataFrame with cash rent records
        """
        if rent_type not in self.RENT_TYPES:
            raise ValueError(f"rent_type must be one of {list(self.RENT_TYPES.keys())}")

        params = {
            "source_desc": "SURVEY",
            "commodity_desc": "RENT",
            "agg_level_desc": "COUNTY",
            "short_desc": self.RENT_TYPES[rent_type],
        }

        if year:
            params["year"] = year
        if state:
            params["state_alpha"] = state

        console.print(f"[blue]Querying NASS API for {rent_type}...[/blue]")
        result = self._make_request(params)

        if "data" not in result:
            console.print(f"[yellow]No data returned for {rent_type}[/yellow]")
            return pd.DataFrame()

        df = pd.DataFrame(result["data"])
        console.print(f"[green]Retrieved {len(df)} records for {rent_type}[/green]")
        return df

    def download_all_cash_rents(
        self,
        start_year: int = 2008,
        end_year: int = 2024,
        output_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Download all cash rents data for all rent types and years.

        Args:
            start_year: First year to download
            end_year: Last year to download
            output_dir: Optional directory to save raw data

        Returns:
            Combined DataFrame with all cash rents data
        """
        settings = get_settings()
        output_dir = output_dir or settings.raw_dir / "nass_cash_rents"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_data = []

        with Progress() as progress:
            total_queries = len(self.RENT_TYPES)
            task = progress.add_task("[cyan]Downloading cash rents...", total=total_queries)

            for rent_type, short_desc in self.RENT_TYPES.items():
                console.print(f"\n[bold]Downloading {rent_type}[/bold]")

                # Query all years at once (API handles it efficiently)
                params = {
                    "source_desc": "SURVEY",
                    "commodity_desc": "RENT",
                    "agg_level_desc": "COUNTY",
                    "short_desc": short_desc,
                    "year__GE": start_year,
                    "year__LE": end_year,
                }

                try:
                    result = self._make_request(params)

                    if "data" in result and result["data"]:
                        df = pd.DataFrame(result["data"])
                        df["rent_type"] = rent_type

                        # Save raw data
                        raw_file = output_dir / f"{rent_type}.csv"
                        df.to_csv(raw_file, index=False)
                        console.print(f"[green]Saved {len(df)} records to {raw_file}[/green]")

                        all_data.append(df)
                    else:
                        console.print(f"[yellow]No data for {rent_type}[/yellow]")

                except httpx.HTTPError as e:
                    console.print(f"[red]Error downloading {rent_type}: {e}[/red]")

                progress.update(task, advance=1)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined_file = output_dir / "all_cash_rents_raw.csv"
            combined.to_csv(combined_file, index=False)
            console.print(f"\n[bold green]Total: {len(combined)} records saved to {combined_file}[/bold green]")
            return combined

        return pd.DataFrame()


def clean_cash_rents(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize cash rents data.

    Args:
        df: Raw DataFrame from NASS API

    Returns:
        Cleaned DataFrame with standardized columns and values
    """
    if df.empty:
        return df

    # Select and rename columns
    columns_map = {
        "state_fips_code": "state_fips",
        "county_code": "county_fips_suffix",
        "state_alpha": "state_abbr",
        "state_name": "state_name",
        "county_name": "county_name",
        "year": "year",
        "Value": "rent_value",
        "CV (%)": "cv_percent",
        "rent_type": "rent_type",
    }

    # Filter to needed columns
    available_cols = [c for c in columns_map.keys() if c in df.columns]
    df_clean = df[available_cols].copy()
    df_clean = df_clean.rename(columns={k: v for k, v in columns_map.items() if k in available_cols})

    # Create full FIPS code (5 digits)
    df_clean["state_fips"] = df_clean["state_fips"].astype(str).str.zfill(2)
    df_clean["county_fips_suffix"] = df_clean["county_fips_suffix"].astype(str).str.zfill(3)
    df_clean["county_fips"] = df_clean["state_fips"] + df_clean["county_fips_suffix"]

    # Convert year to int
    df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce").astype("Int64")

    # Handle suppressed values (marked with (D), (Z), etc.)
    df_clean["is_suppressed"] = df_clean["rent_value"].astype(str).str.contains(
        r"^\(|^\s*$", regex=True, na=True
    )

    # Convert rent values to numeric (suppressed become NaN)
    df_clean["rent_value"] = pd.to_numeric(
        df_clean["rent_value"].astype(str).str.replace(",", ""),
        errors="coerce"
    )

    # Convert CV to numeric
    if "cv_percent" in df_clean.columns:
        df_clean["cv_percent"] = pd.to_numeric(df_clean["cv_percent"], errors="coerce")

    # Data quality flag
    df_clean["data_source"] = "observed"
    df_clean.loc[df_clean["is_suppressed"], "data_source"] = "suppressed"

    console.print(f"[blue]Cleaned data: {len(df_clean)} records[/blue]")
    console.print(f"[blue]  - Observed values: {(~df_clean['is_suppressed']).sum()}[/blue]")
    console.print(f"[blue]  - Suppressed values: {df_clean['is_suppressed'].sum()}[/blue]")

    return df_clean


def pivot_cash_rents(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot cash rents data from long to wide format.

    Args:
        df: Cleaned DataFrame in long format

    Returns:
        Wide format DataFrame with separate columns for each rent type
    """
    if df.empty:
        return df

    # Keep only non-suppressed observed values for pivoting
    df_observed = df[~df["is_suppressed"]].copy()

    # Pivot to wide format
    pivot = df_observed.pivot_table(
        index=["county_fips", "state_fips", "county_fips_suffix", "state_abbr",
               "state_name", "county_name", "year"],
        columns="rent_type",
        values="rent_value",
        aggfunc="first"  # Should be unique, but just in case
    ).reset_index()

    # Flatten column names
    pivot.columns.name = None

    # Rename rent columns
    rename_map = {
        "cropland_nonirrigated": "cropland_rent_nonirrigated",
        "cropland_irrigated": "cropland_rent_irrigated",
        "pastureland": "pasture_rent",
    }
    pivot = pivot.rename(columns=rename_map)

    console.print(f"[blue]Pivoted data: {len(pivot)} county-year observations[/blue]")

    return pivot
