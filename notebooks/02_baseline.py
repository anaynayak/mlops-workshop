import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from mlops_workshop.features import prepare_features, get_feature_columns, get_target_column
    from mlops_workshop.evaluate import evaluate_model, print_metrics
    from mlops_workshop.train import save_model
    from pathlib import Path
    return mo, pd, train_test_split, RandomForestRegressor, prepare_features, get_feature_columns, get_target_column, evaluate_model, print_metrics, save_model, Path


@app.cell
def _():
    print("# MLOps Workshop - Stage 2: Baseline Model")
    print("")
    print("Build a baseline model to predict trip duration.")


@app.cell
def _(pd, prepare_features):
    print("## Load and Prepare Data")
    _df_raw = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    print(f"Raw data: {len(_df_raw):,} rows (10% sample)")

    df = prepare_features(_df_raw)
    print(f"After filtering: {len(df):,} rows")
    print(f"Removed {len(_df_raw) - len(df):,} invalid trips")
    return (df,)


@app.cell
def _(get_feature_columns, get_target_column):
    print("## Features and Target")
    print("")
    print(f"Features: {get_feature_columns()}")
    print(f"Target: {get_target_column()}")


@app.cell
def _(df, get_feature_columns, get_target_column, train_test_split):
    print("## Train-Test Split")
    X = df[get_feature_columns()]
    y = df[get_target_column()]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train):,} rows")
    print(f"Test set: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


@app.cell
def _(X_train, y_train, RandomForestRegressor):
    print("## Train Baseline Model")
    print("Training Random Forest (n_estimators=100, max_depth=10)...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("✓ Training complete")
    return (model,)


@app.cell
def _(model, X_test, y_test, evaluate_model, print_metrics):
    print("## Evaluate Model")
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    print_metrics(metrics)
    return (metrics,)


@app.cell
def _(model, pd, get_feature_columns):
    print("## Feature Importance")
    _importance = pd.DataFrame({
        "feature": get_feature_columns(),
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(_importance)


@app.cell
def _(model, save_model, Path):
    print("## Save Model")
    Path("models").mkdir(exist_ok=True)
    save_model(model, "models/baseline_model.joblib")
    print("✓ Model saved to models/baseline_model.joblib")


@app.cell
def _(metrics):
    print("## Summary")
    print("Baseline model performance:")
    print(f"  RMSE: {metrics['rmse_minutes']:.2f} minutes")
    print(f"  R²: {metrics['r2']:.4f}")
    print("")
    print("Next: Stage 3 - Tune hyperparameters and track with MLflow")


if __name__ == "__main__":
    app.run()
