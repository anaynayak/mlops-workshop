#!/usr/bin/env python
"""Create the notebook sample from the downloaded dataset.

If the source is already about sample size (e.g. the released ~628k-row
sample), take the first TARGET_ROWS deterministically. If it is much larger
(the full ~21M-row dataset), draw a random sample instead so we don't bias
toward the earliest trips.
"""

import pandas as pd
from pathlib import Path

TARGET_ROWS = 600_000


def main():
    data_path = Path("data/raw/nyc_taxi.parquet")
    sample_path = Path("data/raw/nyc_taxi_sample.parquet")

    if sample_path.exists():
        print(f"Sample already exists: {sample_path}")
        return

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Run: make data")
        return

    print("Loading dataset...")
    df = pd.read_parquet(data_path)
    print(f"Dataset: {len(df):,} rows")

    if len(df) <= TARGET_ROWS * 2:
        print(f"Taking first {TARGET_ROWS:,} rows...")
        sample = df.head(TARGET_ROWS)
    else:
        print(f"Random-sampling {TARGET_ROWS:,} rows...")
        sample = df.sample(n=TARGET_ROWS, random_state=42)

    print(f"Sample: {len(sample):,} rows")
    sample.to_parquet(sample_path, index=False)
    print(f"Saved to: {sample_path}")


if __name__ == "__main__":
    main()
