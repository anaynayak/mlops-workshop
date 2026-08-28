import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from mlops_workshop.features import prepare_features, get_feature_columns, get_target_column
    from mlops_workshop.evaluate import evaluate_model

    return (
        RandomForestRegressor,
        evaluate_model,
        get_feature_columns,
        get_target_column,
        mlflow,
        pd,
        prepare_features,
        train_test_split,
    )


@app.cell
def _():
    print("# Stage 3: Experimentation")
    print("")
    print("Track every run with MLflow — params, metrics, and the model artifact.")
    return


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
    return


@app.cell
def _(df, get_feature_columns, get_target_column, train_test_split):
    X = df[get_feature_columns()]
    y = df[get_target_column()]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_test, X_train, y_test, y_train


@app.cell
def _(RandomForestRegressor, X_test, X_train, evaluate_model, mlflow, y_test, y_train):
    print("### Run 1: n_estimators=50, max_depth=5")

    with mlflow.start_run(run_name="rf_shallow"):
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 5)

        _model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)
        _metrics = evaluate_model(y_test, _model.predict(X_test))

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, name="model")  # artifact only — register in Stage 4

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")
    return


@app.cell
def _(RandomForestRegressor, X_test, X_train, evaluate_model, mlflow, y_test, y_train):
    print("### Run 2: n_estimators=100, max_depth=10")

    with mlflow.start_run(run_name="rf_baseline"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        _model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)
        _metrics = evaluate_model(y_test, _model.predict(X_test))

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, name="model")  # artifact only — register in Stage 4

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")
    return


@app.cell
def _(RandomForestRegressor, X_test, X_train, evaluate_model, mlflow, y_test, y_train):
    print("### Run 3: n_estimators=100, max_depth=15")

    with mlflow.start_run(run_name="rf_deeper"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 15)

        _model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)
        _metrics = evaluate_model(y_test, _model.predict(X_test))

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, name="model")  # artifact only — register in Stage 4

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")
    return


@app.cell
def _():
    print("## View Results")
    print("  make mlflow")
    return


@app.cell
def _():
    print("## Summary")
    print("MLflow tracked every run:")
    print("  - Parameters & metrics, comparable in one place")
    print("  - The model artifact for each run")
    print("")
    print("Nothing is deployed yet — choosing and promoting a model")
    print("is Stage 4 (the Model Registry).")
    return


if __name__ == "__main__":
    app.run()
