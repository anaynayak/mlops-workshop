"""Helpers for the Feast feature-store workshop demo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from mlops_workshop.features import (
    add_request_time_features,
    build_route_id,
    filter_valid_trips,
    get_feature_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO_PATH = REPO_ROOT / "feature_repo"
FEATURE_DATA_DIR = FEATURE_REPO_PATH / "data"
ROUTE_STATS_PATH = FEATURE_DATA_DIR / "route_stats.parquet"
RAW_SAMPLE_PATH = REPO_ROOT / "data" / "raw" / "nyc_taxi_sample.parquet"
FEATURE_MODEL_PATH = REPO_ROOT / "models" / "feature_store_model.joblib"

FEATURE_VIEW_NAME = "route_stats"
FEATURE_SERVICE_NAME = "route_features_v1"
ROUTE_FEATURE_NAME = "route_avg_duration_24h"
FEATURE_STORE_MAX_ROWS = 100_000


def get_feature_store() -> FeatureStore:
    """Return a Feast store pointed at the local workshop repo."""
    return FeatureStore(repo_path=str(FEATURE_REPO_PATH))


def get_feature_store_feature_columns() -> list[str]:
    """Return the model columns used by the feature-store demo."""
    return [*get_feature_columns(), ROUTE_FEATURE_NAME]


def sample_demo_data(raw_df: pd.DataFrame, max_rows: int = FEATURE_STORE_MAX_ROWS) -> pd.DataFrame:
    """Keep the Feast demo small enough to run live in the workshop."""
    if len(raw_df) <= max_rows:
        return raw_df.copy()

    return (
        raw_df.sample(n=max_rows, random_state=42)
        .sort_values("pickup_datetime")
        .reset_index(drop=True)
    )


def build_route_statistics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a historical 24-hour average trip duration per route."""
    df = filter_valid_trips(raw_df)
    df = df.dropna(subset=["pickup_datetime", "PULocationID", "DOLocationID", "trip_time"])
    df = df.copy()
    df["route_id"] = build_route_id(df)
    df = df.sort_values(["route_id", "pickup_datetime"])

    route_stats = (
        df.groupby("route_id")
        .rolling("24h", on="pickup_datetime", closed="left")["trip_time"]
        .mean()
        .reset_index(name=ROUTE_FEATURE_NAME)
    )
    route_stats = route_stats.dropna(subset=[ROUTE_FEATURE_NAME]).rename(
        columns={"pickup_datetime": "event_timestamp"}
    )
    route_stats["event_timestamp"] = pd.to_datetime(route_stats["event_timestamp"]).dt.tz_localize(
        None
    )
    route_stats["created_timestamp"] = route_stats["event_timestamp"]
    return route_stats[
        ["route_id", "event_timestamp", ROUTE_FEATURE_NAME, "created_timestamp"]
    ].reset_index(drop=True)


def write_route_statistics(
    raw_df: pd.DataFrame,
    output_path: Path = ROUTE_STATS_PATH,
) -> pd.DataFrame:
    """Persist the route statistics that Feast reads from parquet."""
    route_stats = build_route_statistics(raw_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    route_stats.to_parquet(output_path, index=False)
    return route_stats


def apply_feature_repo(store: FeatureStore | None = None) -> FeatureStore:
    """Register the workshop feature definitions with Feast."""
    feature_store = store or get_feature_store()
    definitions_path = FEATURE_REPO_PATH / "definitions.py"
    spec = importlib.util.spec_from_file_location("workshop_feature_repo_definitions", definitions_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Feast definitions from {definitions_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    feature_store.apply(module.get_feature_store_objects())
    return feature_store


def materialize_features(
    route_stats: pd.DataFrame,
    store: FeatureStore | None = None,
) -> FeatureStore:
    """Load historical route features into the Feast online store."""
    if route_stats.empty:
        raise ValueError("No route statistics were generated; cannot materialize features")

    feature_store = store or get_feature_store()
    online_df = route_stats[
        ["route_id", "event_timestamp", ROUTE_FEATURE_NAME, "created_timestamp"]
    ].copy()
    feature_store.write_to_online_store(feature_view_name=FEATURE_VIEW_NAME, df=online_df)
    return feature_store


def bootstrap_feature_store(raw_df: pd.DataFrame) -> tuple[FeatureStore, pd.DataFrame]:
    """Write feature data, register definitions, and materialize the online store."""
    route_stats = write_route_statistics(raw_df)
    store = apply_feature_repo()
    materialize_features(route_stats=route_stats, store=store)
    return store, route_stats


def build_training_entities(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the entity dataframe Feast joins against for training."""
    df = add_request_time_features(filter_valid_trips(raw_df))
    df = df.dropna(
        subset=["pickup_datetime", "trip_miles", "PULocationID", "DOLocationID", "trip_time"]
    ).copy()
    df["route_id"] = build_route_id(df)
    df["PULocationID"] = df["PULocationID"].astype(int)
    df["DOLocationID"] = df["DOLocationID"].astype(int)

    return df[
        [
            "route_id",
            "pickup_datetime",
            "trip_miles",
            "PULocationID",
            "DOLocationID",
            "pickup_hour",
            "day_of_week",
            "trip_time",
        ]
    ].rename(columns={"pickup_datetime": "event_timestamp"}).assign(
        event_timestamp=lambda frame: pd.to_datetime(frame["event_timestamp"]).dt.tz_localize(None)
    )


def build_training_dataframe(
    raw_df: pd.DataFrame,
    store: FeatureStore | None = None,
) -> pd.DataFrame:
    """Fetch a point-in-time correct training set from Feast."""
    feature_store = store or get_feature_store()
    entity_df = build_training_entities(raw_df)
    training_df = feature_store.get_historical_features(
        entity_df=entity_df,
        features=[f"{FEATURE_VIEW_NAME}:{ROUTE_FEATURE_NAME}"],
        full_feature_names=False,
    ).to_df()

    if ROUTE_FEATURE_NAME not in training_df.columns:
        raise ValueError(f"Feast training set is missing '{ROUTE_FEATURE_NAME}'")

    return training_df.dropna(subset=[ROUTE_FEATURE_NAME]).reset_index(drop=True)


def build_request_dataframe(
    trip_miles: float,
    pickup_location_id: int,
    dropoff_location_id: int,
    pickup_datetime: pd.Timestamp,
    route_avg_duration_24h: float,
) -> pd.DataFrame:
    """Build the single-row dataframe used by the serving endpoint."""
    pickup_ts = pd.Timestamp(pickup_datetime)
    return pd.DataFrame(
        [
            {
                "trip_miles": trip_miles,
                "PULocationID": pickup_location_id,
                "DOLocationID": dropoff_location_id,
                "pickup_hour": pickup_ts.hour,
                "day_of_week": pickup_ts.dayofweek,
                ROUTE_FEATURE_NAME: route_avg_duration_24h,
            }
        ]
    )


def get_online_route_features(
    pickup_location_id: int,
    dropoff_location_id: int,
    store: FeatureStore | None = None,
) -> dict[str, float | str | None]:
    """Fetch the latest online route feature for a serving request."""
    feature_store = store or get_feature_store()
    route_id = f"{int(pickup_location_id)}_{int(dropoff_location_id)}"
    response = feature_store.get_online_features(
        features=feature_store.get_feature_service(FEATURE_SERVICE_NAME),
        entity_rows=[{"route_id": route_id}],
        full_feature_names=False,
    ).to_dict()
    values = response.get(ROUTE_FEATURE_NAME, [])
    return {
        "route_id": route_id,
        ROUTE_FEATURE_NAME: values[0] if values else None,
    }
