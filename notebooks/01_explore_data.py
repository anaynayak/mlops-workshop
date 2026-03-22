import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    return np, pd,


@app.cell
def _(pd):
    df = pd.read_parquet("data/raw/nyc_taxi.parquet")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return (df,)


@app.cell
def _(df):
    print("Columns:")
    print(df.columns.tolist())


@app.cell
def _(df):
    print("Dtypes:")
    print(df.dtypes)


@app.cell
def _(df):
    print("Sample:")
    print(df.head(10))


@app.cell
def _(df):
    print("Null counts:")
    print(df.isnull().sum())


@app.cell
def _(df):
    print("Target variable: trip_time (seconds)")
    print(f"Mean: {df['trip_time'].mean() / 60:.1f} min")
    print(f"Median: {df['trip_time'].median() / 60:.1f} min")
    print(f"Min: {df['trip_time'].min()} seconds")
    print(f"Max: {df['trip_time'].max() / 3600:.1f} hours")


@app.cell
def _(df):
    print("Trip time distribution (minutes):")
    print(df['trip_time'].describe() / 60)


@app.cell
def _(df):
    print("Potential feature: trip_miles")
    print(df['trip_miles'].describe())


@app.cell
def _(df):
    print("Correlation: trip_miles vs trip_time")
    corr = df['trip_miles'].corr(df['trip_time'])
    print(f"Correlation: {corr:.3f}")


@app.cell
def _(df):
    print("Potential feature: PULocationID (pickup location)")
    print(f"Unique locations: {df['PULocationID'].nunique()}")
    print("\nTop 10 pickup locations:")
    print(df['PULocationID'].value_counts().head(10))


@app.cell
def _(df):
    print("Potential feature: DOLocationID (dropoff location)")
    print(f"Unique locations: {df['DOLocationID'].nunique()}")


@app.cell
def _(df):
    print("Potential feature: pickup hour of day")
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    print("\nTrips by hour:")
    print(df['pickup_hour'].value_counts().sort_index())


@app.cell
def _(df):
    print("Potential feature: day of week")
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    print("\nTrips by day (0=Monday):")
    print(df['day_of_week'].value_counts().sort_index())


@app.cell
def _(df):
    print("Data quality check: trip_time outliers")
    very_short = (df['trip_time'] < 60).sum()
    very_long = (df['trip_time'] > 7200).sum()
    print(f"Trips < 1 min: {very_short:,} ({very_short/len(df)*100:.2f}%)")
    print(f"Trips > 2 hours: {very_long:,} ({very_long/len(df)*100:.2f}%)")


@app.cell
def _(df):
    print("Data quality check: trip_miles outliers")
    zero_miles = (df['trip_miles'] == 0).sum()
    very_long_miles = (df['trip_miles'] > 100).sum()
    print(f"Trips with 0 miles: {zero_miles:,} ({zero_miles/len(df)*100:.2f}%)")
    print(f"Trips > 100 miles: {very_long_miles:,} ({very_long_miles/len(df)*100:.2f}%)")


@app.cell
def _(df):
    print("Summary: features to use")
    print("- trip_miles: strong predictor of trip_time")
    print("- PULocationID, DOLocationID: location matters")
    print("- pickup_hour: traffic patterns")
    print("- day_of_week: weekday vs weekend")
    print("\nFilter criteria:")
    print("- trip_time between 60s and 7200s (1 min to 2 hours)")
    print("- trip_miles > 0")


if __name__ == "__main__":
    app.run()
