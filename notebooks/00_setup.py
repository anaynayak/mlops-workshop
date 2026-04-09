import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import sys
    from pathlib import Path

    return Path, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # MLOps Workshop — Stage 0: Setup
    """)
    return


@app.cell
def _(Path, mo):
    _packages = ["pandas", "sklearn", "mlflow", "plotly"]
    _dirs = ["data/raw", "models", "output", "mlruns", "src/mlops_workshop"]
    _checks = "**Environment:** " + " | ".join(
        f"{'✓' if __import__('importlib').util.find_spec(p) else '✗'} {p}"
        for p in _packages
    )
    _structure = "**Project:** " + " | ".join(
        f"{'✓' if Path(d).exists() else '✗'} {d}"
        for d in _dirs
    )
    mo.md(f"{_checks}\n\n{_structure}")
    return


@app.cell
def _(Path, pd):
    _data_path = Path("data/raw/nyc_taxi_sample.parquet")
    if _data_path.exists():
        df = pd.read_parquet(_data_path)
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    else:
        print("Data not found. Run: make data && make sample")
        df = None
    return (df,)


@app.cell
def _(df):
    df.head(10000)
    return


@app.cell
def _(df, mo):
    mo.md(f"""
    **Target:** `trip_time` (trip duration in seconds)  \nMean: **{df['trip_time'].mean() / 60:.1f} min** | Median: **{df['trip_time'].median() / 60:.1f} min**
    """)
    return


if __name__ == "__main__":
    app.run()
