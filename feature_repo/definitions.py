"""Feast definitions for the workshop feature-store demo."""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import Float64

FEATURE_REPO_PATH = Path(__file__).resolve().parent
ROUTE_STATS_PATH = FEATURE_REPO_PATH / "data" / "route_stats.parquet"

route = Entity(
    name="route_id",
    join_keys=["route_id"],
    value_type=ValueType.STRING,
    description="Pickup and dropoff route identifier",
)

route_stats_source = FileSource(
    name="route_stats_source",
    path=str(ROUTE_STATS_PATH),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

route_stats = FeatureView(
    name="route_stats",
    entities=[route],
    ttl=timedelta(days=2),
    schema=[Field(name="route_avg_duration_24h", dtype=Float64)],
    source=route_stats_source,
    online=True,
    offline=True,
    description="24-hour rolling average trip duration per route",
)

route_features_v1 = FeatureService(
    name="route_features_v1",
    features=[route_stats],
    description="Latest route statistics for model serving",
)


def get_feature_store_objects():
    """Return Feast objects to apply for the workshop repo."""
    return [route, route_stats, route_features_v1]
