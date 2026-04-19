import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from mlops_workshop.features import prepare_features, get_feature_columns, get_target_column
    from mlops_workshop.evaluate import evaluate_model, print_metrics
    from pathlib import Path
    return mo, pd, mlflow, MlflowClient, train_test_split, RandomForestRegressor, prepare_features, get_feature_columns, get_target_column, evaluate_model, print_metrics, Path


@app.cell
def _():
    print("# Stage 3: Experimentation")
    print("")
    print("Track experiments and register models with MLflow.")


@app.cell
def _(pd, prepare_features):
    _df_raw = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    df = prepare_features(_df_raw)
    print(f"Dataset: {len(df):,} rows (10% sample)")
    return (df,)


@app.cell
def _(mlflow):
    print("## Configure MLflow")
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("nyc-taxi-duration")
    print("✓ Tracking URI: sqlite:///mlruns/mlflow.db")
    print("✓ Experiment: nyc-taxi-duration")


@app.cell
def _(df, get_feature_columns, get_target_column, train_test_split):
    X = df[get_feature_columns()]
    y = df[get_target_column()]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


@app.cell
def _(X_train, y_train, X_test, y_test, mlflow, RandomForestRegressor, evaluate_model):
    print("### Run 1: n_estimators=50, max_depth=5")

    with mlflow.start_run(run_name="rf_shallow"):
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 5)

        _model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)
        _metrics = evaluate_model(y_test, _model.predict(X_test))

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, name="model", registered_model_name="nyc-taxi-model")

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")


@app.cell
def _(X_train, y_train, X_test, y_test, mlflow, RandomForestRegressor, evaluate_model):
    print("### Run 2: n_estimators=100, max_depth=10")

    with mlflow.start_run(run_name="rf_baseline"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        _model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)
        _metrics = evaluate_model(y_test, _model.predict(X_test))

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, name="model", registered_model_name="nyc-taxi-model")

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")


@app.cell
def _(X_train, y_train, X_test, y_test, mlflow, RandomForestRegressor, evaluate_model):
    print("### Run 3: n_estimators=100, max_depth=15")

    with mlflow.start_run(run_name="rf_deeper"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 15)

        _model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)
        _metrics = evaluate_model(y_test, _model.predict(X_test))

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, name="model", registered_model_name="nyc-taxi-model")

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")


@app.cell
def _(mlflow, MlflowClient):
    print("## Set Challenger Alias")
    print("The best new model becomes the challenger. Champion stays as-is.\n")
    _client = MlflowClient()

    _versions = _client.search_model_versions("name='nyc-taxi-model'")
    print(f"Registered {len(_versions)} model versions:")

    # Find the best new version (lowest RMSE)
    _best = None
    _best_rmse = float("inf")
    for _v in _versions:
        _run = _client.get_run(_v.run_id)
        _rmse = _run.data.metrics.get("rmse", float("inf"))
        print(f"  Version {_v.version}: RMSE={_rmse:.2f}s")
        if _rmse < _best_rmse:
            _best_rmse = _rmse
            _best = _v

    if _best:
        _client.set_registered_model_alias("nyc-taxi-model", "challenger", int(_best.version))
        print(f"\n✓ Challenger: Version {_best.version} (RMSE={_best_rmse:.2f}s)")
        print("  Champion unchanged — promote only if challenger beats it.")


@app.cell
def _():
    print("## View Results")
    print("  make mlflow")


@app.cell
def _():
    print("## Summary")
    print("MLflow tracks:")
    print("  - Parameters & Metrics for each run")
    print("  - Model versions in the registry")
    print("  - Champion/Challenger aliases for deployment")
    print("")
    print("Next: Stage 4 - Inference")


if __name__ == "__main__":
    app.run()
