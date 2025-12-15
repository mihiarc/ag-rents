"""Spatial interpolation model for filling missing cash rent values."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from rich.console import Console

from ag_rents.config import get_settings

console = Console()


def calculate_distance_matrix(
    df: pd.DataFrame,
    lat_col: str = "centroid_lat",
    lon_col: str = "centroid_lon",
) -> np.ndarray:
    """Calculate pairwise distance matrix between counties.

    Args:
        df: DataFrame with county centroids
        lat_col: Column name for latitude
        lon_col: Column name for longitude

    Returns:
        Distance matrix in kilometers
    """
    coords = df[[lat_col, lon_col]].values

    # Haversine distance formula
    lat1 = np.radians(coords[:, 0])[:, np.newaxis]
    lat2 = np.radians(coords[:, 0])[np.newaxis, :]
    lon1 = np.radians(coords[:, 1])[:, np.newaxis]
    lon2 = np.radians(coords[:, 1])[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth's radius in km

    return c * r


def inverse_distance_weighted(
    df: pd.DataFrame,
    value_col: str,
    lat_col: str = "centroid_lat",
    lon_col: str = "centroid_lon",
    power: float = 2.0,
    max_distance_km: float = 200.0,
    min_neighbors: int = 3,
) -> pd.Series:
    """Fill missing values using inverse distance weighting.

    Args:
        df: DataFrame with values and coordinates
        value_col: Column name for values to interpolate
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        power: Power parameter for IDW (higher = more local)
        max_distance_km: Maximum distance to consider neighbors
        min_neighbors: Minimum number of neighbors to use

    Returns:
        Series with filled values
    """
    values = df[value_col].copy()
    has_value = values.notna()

    if has_value.all():
        return values

    # Calculate distances
    distances = calculate_distance_matrix(df, lat_col, lon_col)

    # For each missing value, interpolate from neighbors
    missing_idx = np.where(~has_value)[0]
    observed_idx = np.where(has_value)[0]

    for idx in missing_idx:
        # Get distances to observed counties
        dist_to_observed = distances[idx, observed_idx]

        # Filter by max distance
        within_range = dist_to_observed <= max_distance_km

        if within_range.sum() < min_neighbors:
            # Use closest neighbors if not enough within range
            closest = np.argsort(dist_to_observed)[:min_neighbors]
            within_range = np.zeros_like(within_range, dtype=bool)
            within_range[closest] = True

        neighbor_distances = dist_to_observed[within_range]
        neighbor_values = values.iloc[observed_idx[within_range]].values

        # IDW weights (add small epsilon to avoid division by zero)
        weights = 1 / (neighbor_distances + 0.001) ** power
        weights = weights / weights.sum()

        values.iloc[idx] = np.sum(weights * neighbor_values)

    return values


def train_rent_prediction_model(
    df_train: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[GradientBoostingRegressor, StandardScaler, float]:
    """Train a model to predict cash rents from soil/location features.

    Args:
        df_train: Training DataFrame (counties with observed rents)
        target_col: Column name for target variable (rent)
        feature_cols: List of feature column names

    Returns:
        Tuple of (trained model, scaler, cross-validation R²)
    """
    # Remove rows with missing values
    df_complete = df_train.dropna(subset=[target_col] + feature_cols)

    if len(df_complete) < 50:
        console.print("[yellow]Warning: Very few training samples[/yellow]")

    X = df_complete[feature_cols].values
    y = df_complete[target_col].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train gradient boosting model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    cv_r2 = cv_scores.mean()

    console.print(f"[blue]Cross-validation R²: {cv_r2:.3f} (+/- {cv_scores.std():.3f})[/blue]")

    # Fit on full training data
    model.fit(X_scaled, y)

    return model, scaler, cv_r2


def predict_missing_rents(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model: GradientBoostingRegressor | None = None,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """Predict missing rent values using trained model.

    Args:
        df: Full DataFrame including missing values
        target_col: Column name for rent values
        feature_cols: List of feature column names
        model: Pre-trained model (will train if None)
        scaler: Pre-fitted scaler (will fit if None)

    Returns:
        DataFrame with predicted values and uncertainty estimates
    """
    df = df.copy()
    has_rent = df[target_col].notna()

    # Train model if not provided
    if model is None or scaler is None:
        df_train = df[has_rent].copy()
        model, scaler, cv_r2 = train_rent_prediction_model(
            df_train, target_col, feature_cols
        )

    # Identify counties needing prediction
    df_missing = df[~has_rent].copy()

    if df_missing.empty:
        console.print("[blue]No missing values to predict[/blue]")
        df["prediction_se"] = np.nan
        df["data_source"] = "observed"
        return df

    # Check for complete features
    df_missing_complete = df_missing.dropna(subset=feature_cols)

    if df_missing_complete.empty:
        console.print("[yellow]Missing values have incomplete features[/yellow]")
        df["prediction_se"] = np.nan
        df["data_source"] = np.where(df[target_col].isna(), "missing", "observed")
        return df

    # Predict
    X_missing = df_missing_complete[feature_cols].values
    X_missing_scaled = scaler.transform(X_missing)

    predictions = model.predict(X_missing_scaled)

    # Estimate prediction uncertainty using model variance
    # (simplified: use residual standard error from training)
    df_train = df[has_rent].dropna(subset=feature_cols)
    X_train = scaler.transform(df_train[feature_cols].values)
    train_predictions = model.predict(X_train)
    residuals = df_train[target_col].values - train_predictions
    prediction_se = np.std(residuals)

    # Fill in predictions
    df.loc[df_missing_complete.index, target_col] = predictions
    df.loc[df_missing_complete.index, "prediction_se"] = prediction_se
    df["data_source"] = np.where(has_rent, "observed", "predicted")

    console.print(f"[green]Predicted {len(predictions)} missing values[/green]")
    console.print(f"[blue]Prediction SE: ${prediction_se:.2f}/acre[/blue]")

    return df


def interpolate_cash_rents(
    df_rents: pd.DataFrame,
    df_counties: pd.DataFrame,
    df_nccpi: pd.DataFrame | None = None,
    rent_cols: list[str] = ["cropland_rent_nonirrigated", "pasture_rent"],
    method: Literal["idw", "model", "hybrid"] = "hybrid",
) -> pd.DataFrame:
    """Interpolate missing cash rent values using spatial methods.

    Args:
        df_rents: Cash rents DataFrame with county_fips and rent columns
        df_counties: County boundaries with centroids
        df_nccpi: Optional NCCPI soil productivity data
        rent_cols: List of rent columns to interpolate
        method: Interpolation method ('idw', 'model', or 'hybrid')

    Returns:
        DataFrame with interpolated rent values
    """
    console.print(f"[bold]Interpolating missing cash rents using {method} method[/bold]")

    # Merge rents with county data
    df = df_counties[["county_fips", "state_fips", "county_name",
                      "centroid_lat", "centroid_lon", "land_area_sqmi"]].copy()
    df = df.merge(df_rents, on="county_fips", how="left", suffixes=("", "_rent"))

    # Add NCCPI if available
    if df_nccpi is not None:
        df = df.merge(
            df_nccpi[["county_fips", "nccpi"]],
            on="county_fips",
            how="left"
        )

    # Feature columns for model-based prediction
    feature_cols = ["centroid_lat", "centroid_lon"]
    if "nccpi" in df.columns:
        feature_cols.append("nccpi")
    if "land_area_sqmi" in df.columns:
        feature_cols.append("land_area_sqmi")

    for rent_col in rent_cols:
        if rent_col not in df.columns:
            continue

        missing_count = df[rent_col].isna().sum()
        console.print(f"\n[blue]{rent_col}: {missing_count} missing values[/blue]")

        if missing_count == 0:
            continue

        if method == "idw":
            df[rent_col] = inverse_distance_weighted(df, rent_col)
            df["data_source"] = np.where(
                df[rent_col].notna() & df_rents.set_index("county_fips").reindex(df["county_fips"])[rent_col].isna(),
                "predicted_idw",
                "observed"
            )

        elif method == "model":
            df = predict_missing_rents(df, rent_col, feature_cols)

        elif method == "hybrid":
            # First use model for counties with complete features
            df_complete = df.dropna(subset=feature_cols)
            if len(df_complete) > 100:
                df = predict_missing_rents(df, rent_col, feature_cols)

            # Then use IDW for remaining
            still_missing = df[rent_col].isna().sum()
            if still_missing > 0:
                console.print(f"[blue]Using IDW for {still_missing} remaining values[/blue]")
                idw_values = inverse_distance_weighted(df, rent_col)
                df.loc[df[rent_col].isna(), rent_col] = idw_values[df[rent_col].isna()]
                df.loc[df["data_source"] == "missing", "data_source"] = "predicted_idw"

    return df


def create_output_dataset(
    df_rents: pd.DataFrame,
    df_counties: pd.DataFrame,
    df_nccpi: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Create the final output dataset with interpolated values.

    Args:
        df_rents: Processed cash rents data
        df_counties: County boundaries
        df_nccpi: Optional NCCPI data
        output_dir: Directory to save output

    Returns:
        Final output DataFrame
    """
    settings = get_settings()
    output_dir = output_dir or settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Interpolate missing values
    df_output = interpolate_cash_rents(
        df_rents, df_counties, df_nccpi,
        rent_cols=["cropland_rent_nonirrigated", "cropland_rent_irrigated", "pasture_rent"],
        method="hybrid"
    )

    # Select and order output columns
    output_cols = [
        "state_fips",
        "county_fips",
        "county_name",
        "year",
        "cropland_rent_nonirrigated",
        "cropland_rent_irrigated",
        "pasture_rent",
        "data_source",
    ]

    if "nccpi" in df_output.columns:
        output_cols.append("nccpi")
    if "prediction_se" in df_output.columns:
        output_cols.append("prediction_se")

    # Filter to available columns
    output_cols = [c for c in output_cols if c in df_output.columns]
    df_final = df_output[output_cols].copy()

    # Save outputs
    csv_file = output_dir / "county_cash_rents.csv"
    parquet_file = output_dir / "county_cash_rents.parquet"

    df_final.to_csv(csv_file, index=False)
    df_final.to_parquet(parquet_file, index=False)

    console.print(f"\n[bold green]Output saved to:[/bold green]")
    console.print(f"  CSV: {csv_file}")
    console.print(f"  Parquet: {parquet_file}")

    # Summary statistics
    console.print(f"\n[bold]Output Summary:[/bold]")
    console.print(f"  Total counties: {len(df_final)}")

    if "data_source" in df_final.columns:
        source_counts = df_final["data_source"].value_counts()
        for source, count in source_counts.items():
            console.print(f"  - {source}: {count}")

    return df_final
