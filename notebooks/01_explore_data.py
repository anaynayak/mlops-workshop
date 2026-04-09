import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px

    return mo, pd, px


@app.cell
def _(pd):
    df = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns (10% sample)")
    return (df,)


@app.cell
def _(mo):
    mo.md("""
    **References:**
    - [NYC Taxi Data Dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
    - [Trip Record User Guide](https://www.nyc.gov/assets/tlc/downloads/pdf/trip_record_user_guide.pdf)
    """)
    return


@app.cell
def _(df):
    print("Dtypes:")
    df.dtypes
    return


@app.cell
def _(df):
    print("Sample:")
    df.head(100000)
    return


@app.cell
def _(df):
    nulls = df.isnull().sum()
    nulls[nulls > 0] if nulls.sum() > 0 else "No nulls found"
    return


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Target: `trip_time` (seconds)
    Mean: **{df['trip_time'].mean() / 60:.1f} min** | Median: **{df['trip_time'].median() / 60:.1f} min** | Min: **{df['trip_time'].min()}s** | Max: **{df['trip_time'].max() / 3600:.1f} hrs**
    """)
    return


@app.cell
def _(df):
    corr = df['trip_miles'].corr(df['trip_time'])
    print(f"Correlation (trip_miles vs trip_time): {corr:.3f}")
    return


@app.cell
def _(df):
    very_short = (df['trip_time'] < 60).sum()
    very_long = (df['trip_time'] > 7200).sum()
    zero_miles = (df['trip_miles'] == 0).sum()
    print(f"Trips < 1 min: {very_short:,} ({very_short/len(df)*100:.2f}%)")
    print(f"Trips > 2 hours: {very_long:,} ({very_long/len(df)*100:.2f}%)")
    print(f"Trips with 0 miles: {zero_miles:,} ({zero_miles/len(df)*100:.2f}%)")
    return


@app.cell
def _(df):
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return


@app.cell
def _(df, px):
    _df_clean = df[(df['trip_time'] >= 60) & (df['trip_time'] <= 7200)]
    _sample = _df_clean.sample(min(50000, len(_df_clean)), random_state=42)
    px.histogram(_sample, x='trip_time', nbins=50,
                 title='Trip Time Distribution',
                 labels={'trip_time': 'Trip Time (seconds)'})
    return


@app.cell
def _(df, px):
    _hourly = df.groupby('pickup_hour').size().reset_index(name='count')
    _fig = px.bar(_hourly, x='pickup_hour', y='count',
                  title='Trips by Hour of Day',
                  labels={'pickup_hour': 'Hour', 'count': 'Number of Trips'})
    _fig.update_xaxes(dtick=1)
    _fig
    return


@app.cell
def _(df, px):
    _hourly_avg = df.groupby('pickup_hour')['trip_time'].mean().reset_index()
    _hourly_avg['trip_time_min'] = _hourly_avg['trip_time'] / 60
    _fig = px.line(_hourly_avg, x='pickup_hour', y='trip_time_min',
                   title='Average Trip Time by Hour',
                   labels={'pickup_hour': 'Hour', 'trip_time_min': 'Avg Trip Time (min)'})
    _fig.update_xaxes(dtick=1)
    _fig
    return


@app.cell
def _(df, px):
    _df_clean = df[(df['trip_time'] >= 60) & (df['trip_time'] <= 7200) & (df['trip_miles'] > 0) & (df['trip_miles'] < 50)]
    _sample = _df_clean.sample(min(10000, len(_df_clean)), random_state=42)
    px.scatter(_sample, x='trip_miles', y='trip_time', color='pickup_hour',
               title='Trip Time vs Trip Miles',
               labels={'trip_miles': 'Trip Miles', 'trip_time': 'Trip Time (seconds)', 'pickup_hour': 'Hour'},
               opacity=0.5)
    return


@app.cell
def _(df, px):
    _days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    _daily = df.groupby('day_of_week').size().reset_index(name='count')
    _daily['day'] = _daily['day_of_week'].map(lambda x: _days[x])
    px.bar(_daily, x='day', y='count',
           title='Trips by Day of Week',
           labels={'day': 'Day', 'count': 'Number of Trips'})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Features to use
    - `trip_miles`: strong predictor of trip_time (corr: 0.78)
    - `PULocationID`, `DOLocationID`: location matters
    - `pickup_hour`: traffic patterns
    - `day_of_week`: weekday vs weekend

    **Filter criteria:** trip_time between 60s–7200s (1 min to 2 hours), trip_miles > 0
    """)
    return


if __name__ == "__main__":
    app.run()
