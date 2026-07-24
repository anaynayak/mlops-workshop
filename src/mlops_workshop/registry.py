"""Model registry helpers for the NYC Taxi workshop.

Thin wrappers over the MLflow Model Registry so participants can focus on the
*workflow* — find a candidate run, register it, promote it, load it, roll back —
without fighting the MLflow API. Each helper does one step; you compose them.

Assumes training runs were logged to the workshop tracking store with the model
under the artifact path ``"model"`` (e.g. ``mlflow.sklearn.log_model(model, "model")``).
"""

import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

# Workshop conventions — same store the training scripts/notebooks write to.
# Defaults to local SQLite; set MLFLOW_TRACKING_URI to run against a hosted
# server (e.g. DagsHub) without code changes — see docs/CLOUD.md.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
EXPERIMENT_NAME = "nyc-taxi-duration"
MODEL_NAME = "nyc-taxi-duration"


def _client() -> MlflowClient:
    """Return an MlflowClient pointed at the workshop tracking store."""
    return MlflowClient(tracking_uri=TRACKING_URI)


def find_runs(
    experiment_name: str = EXPERIMENT_NAME,
    metric: str = "rmse",
    ascending: bool = True,
    max_results: int = 50,
) -> pd.DataFrame:
    """List runs for an experiment, best first, to help pick a candidate.

    Args:
        experiment_name: Experiment to search.
        metric: Metric to sort by (e.g. "rmse", "r2").
        ascending: True for "lower is better" (rmse, mae); False for r2.
        max_results: Maximum number of runs to return.

    Returns:
        DataFrame of runs (one row each) with `run_id`, `params.*`, `metrics.*`.
        Eyeball it and pick the `run_id` you want to promote.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    order = "ASC" if ascending else "DESC"
    return mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} {order}"],
        max_results=max_results,
    )


def register_run(
    run_id: str,
    name: str = MODEL_NAME,
    artifact_path: str = "model",
) -> ModelVersion:
    """Register the model logged in a run as a new version of `name`.

    Args:
        run_id: The run whose logged model you want to register.
        name: Registered model name (created if it doesn't exist).
        artifact_path: Artifact path the model was logged under.

    Returns:
        The newly created ModelVersion (see `.version`).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=name)


def set_alias(alias: str, version: int | str, name: str = MODEL_NAME) -> None:
    """Point an alias (e.g. "champion") at a specific model version.

    Promotion and rollback are both just moving an alias to a different version.

    Args:
        alias: Alias to set, e.g. "champion" (live) or "challenger" (candidate).
        version: Version number to point the alias at.
        name: Registered model name.
    """
    _client().set_registered_model_alias(name=name, alias=alias, version=str(version))


def load_model(alias: str = "champion", name: str = MODEL_NAME):
    """Load the model currently behind an alias.

    Consumers (inference jobs) reference the alias, never a file path or version —
    so promoting/rolling back never requires touching their code.

    Args:
        alias: Alias to resolve, e.g. "champion".
        name: Registered model name.

    Returns:
        The loaded scikit-learn model.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def list_versions(name: str = MODEL_NAME) -> pd.DataFrame:
    """List every registered version of a model and its aliases.

    Args:
        name: Registered model name.

    Returns:
        DataFrame with `version`, `run_id`, `aliases`, `creation_time`.
    """
    client = _client()
    # Aliases live on the registered model as {alias: version}; invert to version -> [aliases].
    alias_map: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(name).aliases.items():
        alias_map.setdefault(str(version), []).append(alias)

    versions = client.search_model_versions(f"name='{name}'")
    return pd.DataFrame(
        [
            {
                "version": int(v.version),
                "run_id": v.run_id,
                "aliases": alias_map.get(str(v.version), []),
                "creation_time": pd.to_datetime(v.creation_timestamp, unit="ms"),
            }
            for v in versions
        ]
    ).sort_values("version", ascending=False, ignore_index=True)
