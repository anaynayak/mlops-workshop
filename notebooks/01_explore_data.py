import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    return Path, mo, pd


@app.cell
def _(Path, pd):
    data_path = Path("data/raw/nyc_taxi.parquet")
    df = pd.read_parquet(data_path)
    f"Loaded {len(df)} rows"
    return (df,)


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


if __name__ == "__main__":
    app.run()
