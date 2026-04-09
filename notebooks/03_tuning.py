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
    return (
        Path,
        RandomForestRegressor,
        evaluate_model,
        get_feature_columns,
        get_target_column,
        pd,
        prepare_features,
        print_metrics,
        save_model,
        train_test_split,
    )


@app.cell
def _():
    print("# Stage 3: Hyperparameter Tuning")
    print("")
    print("Try different configurations and compare results.")


@app.cell
def _(pd, prepare_features):
    print("## Load and Prepare Data")
    _df_raw = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    df = prepare_features(_df_raw)
    print(f"Dataset: {len(df):,} rows (10% sample)")
    return (df,)


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
def _(mo):
    mo.md("## Try Different Hyperparameters")


@app.cell
def _(X_train, y_train, X_test, y_test, RandomForestRegressor, evaluate_model, print_metrics):
    print("### Run 1: n_estimators=50, max_depth=5")
    _model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    _model.fit(X_train, y_train)
    _metrics = evaluate_model(y_test, _model.predict(X_test))
    print_metrics(_metrics)


@app.cell
def _(X_train, y_train, X_test, y_test, RandomForestRegressor, evaluate_model, print_metrics):
    print("### Run 2: n_estimators=100, max_depth=10")
    _model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    _model.fit(X_train, y_train)
    _metrics = evaluate_model(y_test, _model.predict(X_test))
    print_metrics(_metrics)


@app.cell
def _(X_train, y_train, X_test, y_test, RandomForestRegressor, evaluate_model, print_metrics):
    print("### Run 3: n_estimators=100, max_depth=15")
    _model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    _model.fit(X_train, y_train)
    _metrics = evaluate_model(y_test, _model.predict(X_test))
    print_metrics(_metrics)


@app.cell
def _(mo):
    mo.md("""
    ## Problem

    How do you track which configuration produced which result?
    How do you compare runs? How do you reproduce a result?

    **This is where MLflow comes in.**
    """)


if __name__ == "__main__":
    app.run()
