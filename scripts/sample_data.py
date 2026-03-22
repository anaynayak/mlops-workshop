#!/usr/bin/env python
"""Create a 10% sample of the full dataset for faster iteration."""

import pandas as pd
from pathlib import Path


def main():
    data_path = Path("data/raw/nyc_taxi.parquet")
    sample_path = Path("data/raw/nyc_taxi_sample.parquet")

    if sample_path.exists():
        print(f"Sample already exists: {sample_path}")
        return

    if not data_path.exists():
        print(f"Full dataset not found: {data_path}")
        print("Run: make data")
        return

    print(f"Loading full dataset...")
    df = pd.read_parquet(data_path)
    print(f"Full dataset: {len(df):,} rows")

    print("Sampling 10%...")
    sample = df.sample(frac=0.1, random_state=42)
    print(f"Sample: {len(sample):,} rows")

    sample.to_parquet(sample_path, index=False)
    print(f"Saved to: {sample_path}")


if __name__ == "__main__":
    main()
