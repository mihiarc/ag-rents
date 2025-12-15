# Agricultural Land Rents Dataset

County-level annual cash rents for cropland and pastureland across the contiguous United States (2008-2024).

## Overview

This project updates the Lubowski (2002) / Mihiar (2018) methodology for estimating agricultural land rents using modern data sources that provide direct observations of market rents instead of constructed proxies.

### Key Improvements Over Original Methodology

| Component | Original Approach | Updated Approach |
|-----------|------------------|------------------|
| Cropland rent | Constructed from yields × prices - costs | Direct observation (NASS Cash Rents Survey) |
| Pasture rent | Proxy from livestock returns | Direct observation (NASS Cash Rents Survey) |
| Temporal coverage | Point-in-time (1997, 2002) | Annual panel (2008-2024) |
| Missing data | Limited options | Soil productivity (NCCPI) + spatial interpolation |

## Data Sources

1. **NASS Cash Rents Survey** (2008-present) - Direct measurement of land rents via QuickStats API
2. **Census TIGER/Line** - County boundaries for spatial analysis
3. **NRCS Soil Data Access** - NCCPI soil productivity indices for spatial interpolation
4. **gridMET** - Climate data for Ricardian analysis (optional)

## Installation

```bash
# Clone repository
git clone <repository-url>
cd ag-rents

# Create virtual environment and install
uv venv
uv pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
```

## Getting a NASS API Key

1. Visit https://quickstats.nass.usda.gov/api
2. Click "Request API Key"
3. Fill out the form with your email
4. Add the key to your `.env` file:
   ```
   NASS_API_KEY=your_api_key_here
   ```

## Usage

### Step 1: Download Data

```bash
# Download all data sources (requires NASS API key)
uv run python scripts/download_data.py --api-key YOUR_KEY

# Or with API key in environment
export NASS_API_KEY=your_key
uv run python scripts/download_data.py

# Download only county boundaries (no API key needed)
uv run python scripts/download_data.py --skip-nass --skip-nccpi

# Download only NCCPI soil productivity data
uv run python scripts/download_data.py --nccpi-only
```

### Step 2: Process Data

```bash
# Process data and create output files with interpolated values
uv run python scripts/process_data.py
```

### Step 3: Create Visualizations

```bash
# Generate maps and charts
uv run python scripts/create_visualizations.py
```

## Output Files

### Primary Output: `data/output/county_cash_rents_panel.csv`

| Column | Description |
|--------|-------------|
| `state_fips` | 2-digit state FIPS code |
| `county_fips` | 5-digit county FIPS code |
| `county_name` | County name |
| `year` | Data year (2008-2024) |
| `cropland_rent_nonirrigated` | Cash rent for non-irrigated cropland ($/acre) |
| `cropland_rent_irrigated` | Cash rent for irrigated cropland ($/acre) |
| `pasture_rent` | Cash rent for pastureland ($/acre) |
| `data_source` | "observed", "predicted", or "predicted_idw" |
| `nccpi` | National Commodity Crop Productivity Index (0-1) |
| `prediction_se` | Standard error of prediction (if predicted) |

### Additional Outputs

- `county_cash_rents_panel.parquet` - Parquet format for faster loading
- `cash_rents_YYYY.csv` - Yearly files
- `cash_rents_multiyear_avg.csv` - Multi-year averages

## Project Structure

```
ag-rents/
├── data/
│   ├── raw/                    # Downloaded source data
│   │   ├── nass_cash_rents/    # NASS API downloads
│   │   ├── tiger/              # Census county boundaries
│   │   └── gssurgo/            # NCCPI soil data
│   ├── processed/              # Cleaned intermediate data
│   └── output/                 # Final datasets
├── scripts/
│   ├── download_data.py        # Download all data sources
│   ├── process_data.py         # Process and interpolate
│   └── create_visualizations.py # Generate maps and charts
├── src/ag_rents/               # Python package
│   ├── config.py               # Configuration settings
│   ├── nass_client.py          # NASS QuickStats API client
│   ├── boundaries.py           # County boundaries processing
│   ├── soil_productivity.py    # NCCPI data access
│   └── interpolation.py        # Spatial interpolation model
├── figures/                    # Generated visualizations
└── validation/                 # Validation outputs
```

## Spatial Interpolation

For counties missing cash rent data (typically ~10-15% due to disclosure suppression):

1. **Model-based prediction**: Gradient boosting model using:
   - County centroid coordinates (lat/lon)
   - NCCPI soil productivity index
   - Land area

2. **Inverse distance weighting (IDW)**: Fallback for counties without soil data

The `data_source` column indicates whether values are observed or predicted.

## API Reference

```python
from ag_rents.nass_client import NASSClient, clean_cash_rents, pivot_cash_rents
from ag_rents.boundaries import load_or_download_counties
from ag_rents.soil_productivity import load_or_download_nccpi
from ag_rents.interpolation import interpolate_cash_rents

# Download cash rents
client = NASSClient(api_key="YOUR_KEY")
df_raw = client.download_all_cash_rents(start_year=2008, end_year=2024)
df_clean = clean_cash_rents(df_raw)
df_wide = pivot_cash_rents(df_clean)

# Load county boundaries
gdf_counties = load_or_download_counties()

# Load soil productivity
df_nccpi = load_or_download_nccpi()

# Interpolate missing values
df_output = interpolate_cash_rents(df_wide, gdf_counties, df_nccpi)
```

## License

MIT
