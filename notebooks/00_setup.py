import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    print("# MLOps Workshop - Stage 0: Setup")
    print("")
    print("Welcome! This notebook checks your environment and loads the data.")
    return (mo,)


@app.cell
def _():
    print("## Environment Check")
    print("")


@app.cell
def _():
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")


@app.cell
def _():
    print("## Required Packages")
    print("")


@app.cell
def _():
    import pandas
    import sklearn
    import mlflow
    import joblib
    import plotly

    print(f"pandas: {pandas.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"mlflow: {mlflow.__version__}")
    print(f"joblib: {joblib.__version__}")
    print(f"plotly: {plotly.__version__}")


@app.cell
def _():
    print("## Project Structure")
    print("")


@app.cell
def _():
    from pathlib import Path

    print("Project directories:")
    for _p in ["data/raw", "models", "output", "mlruns", "src/mlops_workshop"]:
        _path = Path(_p)
        _exists = "✓" if _path.exists() else "✗"
        print(f"  {_exists} {_p}")


@app.cell
def _():
    print("## Load Data")
    print("")


@app.cell
def _(pd):
    import pandas as pd

    _data_path = Path("data/raw/nyc_taxi.parquet")
    if _data_path.exists():
        df = pd.read_parquet(_data_path)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    else:
        print("✗ Data file not found. Run: make data")
        df = None
    return (df,)


@app.cell
def _(df):
    if df is not None:
        print("## Data Preview")
        print("")
        print(df.head(5))


@app.cell
def _(df):
    if df is not None:
        print("## Target Variable")
        print("")
        print("We'll predict `trip_time` (duration in seconds)")
        print(f"Mean: {df['trip_time'].mean() / 60:.1f} minutes")
        print(f"Median: {df['trip_time'].median() / 60:.1f} minutes")


@app.cell
def _():
    print("## Next Steps")
    print("")
    print("Environment ready! Continue to:")
    print("- Stage 1: Data exploration (01_explore_data.py)")
    print("- Stage 2: Baseline model (02_baseline.py)")


if __name__ == "__main__":
    app.run()
