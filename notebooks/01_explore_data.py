import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    return mo, np, pd, px


@app.cell
def _(pd):
    df = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns (10% sample)")
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
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    print("\nPotential feature: pickup hour of day")
    print(df['pickup_hour'].value_counts().sort_index())


@app.cell
def _(df):
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    print("\nPotential feature: day of week")
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


@app.cell
def _(df, px):
    _df_clean = df[(df['trip_time'] >= 60) & (df['trip_time'] <= 7200)]
    _sample = _df_clean.sample(min(50000, len(_df_clean)), random_state=42)
    _fig = px.histogram(_sample, x='trip_time', nbins=50,
                        title='Trip Time Distribution',
                        labels={'trip_time': 'Trip Time (seconds)'})
    _fig


@app.cell
def _(df, px):
    _hourly = df.groupby('pickup_hour').size().reset_index(name='count')
    _fig = px.bar(_hourly, x='pickup_hour', y='count',
                  title='Trips by Hour of Day',
                  labels={'pickup_hour': 'Hour', 'count': 'Number of Trips'})
    _fig.update_xaxes(dtick=1)
    _fig


@app.cell
def _(df, px):
    _hourly_avg = df.groupby('pickup_hour')['trip_time'].mean().reset_index()
    _hourly_avg['trip_time_min'] = _hourly_avg['trip_time'] / 60
    _fig = px.line(_hourly_avg, x='pickup_hour', y='trip_time_min',
                   title='Average Trip Time by Hour',
                   labels={'pickup_hour': 'Hour', 'trip_time_min': 'Avg Trip Time (min)'})
    _fig.update_xaxes(dtick=1)
    _fig


@app.cell
def _(df, px):
    _df_clean = df[(df['trip_time'] >= 60) & (df['trip_time'] <= 7200) & (df['trip_miles'] > 0) & (df['trip_miles'] < 50)]
    _sample = _df_clean.sample(min(10000, len(_df_clean)), random_state=42)
    _fig = px.scatter(_sample, x='trip_miles', y='trip_time',
                      title='Trip Time vs Trip Miles',
                      labels={'trip_miles': 'Trip Miles', 'trip_time': 'Trip Time (seconds)'},
                      opacity=0.3)
    _fig


@app.cell
def _(df, px):
    _top_pu = df['PULocationID'].value_counts().head(10).reset_index()
    _top_pu.columns = ['PULocationID', 'count']
    _fig = px.bar(_top_pu, x='PULocationID', y='count',
                  title='Top 10 Pickup Locations',
                  labels={'PULocationID': 'Location ID', 'count': 'Number of Trips'})
    _fig


@app.cell
def _(df, px):
    _days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    _daily = df.groupby('day_of_week').size().reset_index(name='count')
    _daily['day'] = _daily['day_of_week'].map(lambda x: _days[x])
    _fig = px.bar(_daily, x='day', y='count',
                  title='Trips by Day of Week',
                  labels={'day': 'Day', 'count': 'Number of Trips'})
    _fig


if __name__ == "__main__":
    app.run()
