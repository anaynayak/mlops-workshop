#!/usr/bin/env python
"""Download the workshop dataset from WORKSHOP_SAMPLE_URL.

Cross-platform replacement for the old `wget` Makefile recipe: works on
Windows, macOS, and Linux with only the Python standard library.
"""

import os
import sys
import urllib.request
from pathlib import Path


def main():
    data_path = Path("data/raw/nyc_taxi.parquet")

    if data_path.exists():
        print(f"Data already exists: {data_path}")
        return

    url = os.environ.get("WORKSHOP_SAMPLE_URL")
    if not url:
        print("Error: WORKSHOP_SAMPLE_URL must be set")
        sys.exit(1)

    data_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset to {data_path} ...")
    urllib.request.urlretrieve(url, data_path)
    print(f"Saved to: {data_path}")


if __name__ == "__main__":
    main()
