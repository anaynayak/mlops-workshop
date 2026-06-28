"""Feature engineering for NYC Taxi trip duration prediction."""

import pandas as pd


def filter_valid_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Keep trips that are valid for the workshop modeling flow."""
    filtered = df.copy()
    filtered = filtered[(filtered["trip_time"] >= 60) & (filtered["trip_time"] <= 7200)]
    filtered = filtered[filtered["trip_miles"] > 0]
    return filtered


def add_request_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add request-time features derived from the pickup timestamp."""
    featured = df.copy()
    featured["pickup_hour"] = featured["pickup_datetime"].dt.hour
    featured["day_of_week"] = featured["pickup_datetime"].dt.dayofweek
    return featured


def build_route_id(df: pd.DataFrame) -> pd.Series:
    """Build the route entity key used by the feature-store demo."""
    return (
        df["PULocationID"].astype("int64").astype(str)
        + "_"
        + df["DOLocationID"].astype("int64").astype(str)
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model training.

    Args:
        df: Raw DataFrame with NYC Taxi data

    Returns:
        DataFrame with engineered features
    """
    df = add_request_time_features(filter_valid_trips(df))

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
