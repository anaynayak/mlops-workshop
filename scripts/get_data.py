#!/usr/bin/env python
"""Download the workshop dataset.

Defaults to the sample published as a release on this repo, so attendees get a
small (~27MB) file with zero configuration. Set WORKSHOP_SAMPLE_URL to override
(e.g. to point at the full dataset or a different sample).

Cross-platform (Windows/macOS/Linux) using only the Python standard library.
"""

import os
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://github.com/anaynayak/mlops-workshop/releases/download/"
    "sample-v1/nyc_taxi_sample.parquet"
)


def main():
    data_path = Path("data/raw/nyc_taxi.parquet")

    if data_path.exists():
        print(f"Data already exists: {data_path}")
        return

    url = os.environ.get("WORKSHOP_SAMPLE_URL", DEFAULT_URL)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, data_path)
    print(f"Saved to: {data_path}")


if __name__ == "__main__":
    main()
