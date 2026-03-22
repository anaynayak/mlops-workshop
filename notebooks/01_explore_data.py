import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return pd,


@app.cell
def _(pd):
    df = pd.read_parquet("data/raw/nyc_taxi.parquet")
    f"Loaded {len(df):,} rows, {len(df.columns)} columns"
    return (df,)


@app.cell
def _(df):
    df.columns.tolist()


@app.cell
def _(df):
    df.dtypes


@app.cell
def _(df):
    df.head(10)


@app.cell
def _(df):
    df[["trip_time", "trip_miles", "base_passenger_fare"]].describe()


@app.cell
def _(df):
    df.isnull().sum()


if __name__ == "__main__":
    app.run()
