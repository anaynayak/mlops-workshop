"""Feature engineering for NYC Taxi trip duration prediction."""

import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model training.

    Args:
        df: Raw DataFrame with NYC Taxi data

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Filter out invalid trips
    df = df[(df["trip_time"] >= 60) & (df["trip_time"] <= 7200)]
    df = df[df["trip_miles"] > 0]

    # Time-based features
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

    # Select features and target
    feature_cols = get_feature_columns()
    target_col = "trip_time"

    result = df[feature_cols + [target_col]].copy()
    result = result.dropna()

    return result


def get_feature_columns() -> list[str]:
    """Return list of feature column names."""
    return ["trip_miles", "PULocationID", "DOLocationID", "pickup_hour", "day_of_week"]


def get_target_column() -> str:
    """Return target column name."""
    return "trip_time"
