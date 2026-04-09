import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    return mo, pd, np, Path


@app.cell
def _():
    print("# Stage 5: Operations")
    print("")
    print("What does it take to run ML in production?")


@app.cell
def _(mo):
    mo.md("""
    ## Testing

    - **Unit tests** — test individual functions (`prepare_features`, `evaluate_model`)
    - **Integration tests** — test the full pipeline end-to-end
    - **Data validation** — check input data quality
    - **Model validation** — check performance thresholds
    """)


@app.cell
def _(mo):
    mo.md("""
    ## CI/CD

    ```yaml
    # .github/workflows/ml-pipeline.yml
    on: push
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - run: make setup
          - run: make test
          - run: make train  # on main branch only
    ```
    """)


@app.cell
def _(pd, np):
    print("## Monitoring: Data Drift Detection")
    _baseline_mean = 1125  # baseline trip_time mean in seconds
    _baseline_std = 792

    _new_sample = np.random.normal(1100, 800, 1000)
    _new_mean = _new_sample.mean()

    _z_score = abs(_new_mean - _baseline_mean) / (_baseline_std / np.sqrt(1000))
    print(f"Baseline mean: {_baseline_mean}s")
    print(f"New data mean: {_new_mean:.0f}s")
    print(f"Z-score: {_z_score:.2f}")
    print(f"Drift detected: {'Yes' if _z_score > 3 else 'No'}")


@app.cell
def _(mo):
    mo.md("""
    ## Retraining Triggers

    - Scheduled (weekly/monthly)
    - Performance degradation detected
    - Data drift detected
    - New feature available
    - Business requirements change
    """)


@app.cell
def _():
    print("## Workshop Summary")
    print("")
    print("Stage 0: Environment setup")
    print("Stage 1: Data exploration")
    print("Stage 2: Baseline model")
    print("Stage 3: Hyperparameter tuning + MLflow")
    print("Stage 4: Batch inference + model registry")
    print("Stage 5: Operations (testing, CI/CD, monitoring)")
    print("")
    print("Next steps:")
    print("  - Add pytest tests to tests/")
    print("  - Set up GitHub Actions CI")
    print("  - Deploy model to production")


if __name__ == "__main__":
    app.run()
