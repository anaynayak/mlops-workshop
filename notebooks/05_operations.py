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
    print("# MLOps Workshop - Stage 5: Operations")
    print("")
    print("Testing, CI/CD, and monitoring concepts for ML systems.")


@app.cell
def _():
    print("## Testing ML Systems")
    print("")
    print("Key test types:")
    print("  1. Unit tests - individual functions")
    print("  2. Integration tests - end-to-end pipelines")
    print("  3. Data validation - input data quality")
    print("  4. Model validation - performance thresholds")


@app.cell
def _():
    print("## Example: Unit Tests")
    print("")
    print("```python")
    print("# tests/test_features.py")
    print("def test_prepare_features_filters_invalid_trips():")
    print("    df = pd.DataFrame({")
    print("        'trip_time': [30, 100, 5000, 8000],")
    print("        'trip_miles': [0, 1, 5, 10],")
    print("        'pickup_datetime': pd.to_datetime(['2024-01-01'] * 4),")
    print("        'PULocationID': [1, 2, 3, 4],")
    print("        'DOLocationID': [5, 6, 7, 8],")
    print("    })")
    print("    result = prepare_features(df)")
    print("    # Should filter out trip_time < 60 and > 7200, and trip_miles == 0")
    print("    assert len(result) == 1  # only the 5000s, 5 mile trip")
    print("```")


@app.cell
def _():
    print("## Example: Model Validation")
    print("")
    print("```python")
    print("# tests/test_model.py")
    print("def test_model_rmse_below_threshold():")
    print("    model = load_model('models/rf_model.joblib')")
    print("    X_test, y_test = load_test_data()")
    print("    predictions = model.predict(X_test)")
    print("    rmse = root_mean_squared_error(y_test, predictions)")
    print("    # Fail if model degrades below threshold")
    print("    assert rmse < 500  # seconds")
    print("```")


@app.cell
def _():
    print("## Data Validation")
    print("")
    print("Check input data before training/inference:")
    print("  - No null values in required columns")
    print("  - Values within expected ranges")
    print("  - No sudden distribution shifts")


@app.cell
def _(pd, np):
    print("### Example: Data Drift Detection")
    _baseline_mean = 1125  # baseline trip_time mean in seconds
    _baseline_std = 792

    # Simulate new data check
    _new_sample = np.random.normal(1100, 800, 1000)
    _new_mean = _new_sample.mean()

    _z_score = abs(_new_mean - _baseline_mean) / (_baseline_std / np.sqrt(1000))
    print(f"Baseline mean: {_baseline_mean}s")
    print(f"New data mean: {_new_mean:.0f}s")
    print(f"Z-score: {_z_score:.2f}")
    print(f"Drift detected: {'Yes' if _z_score > 3 else 'No'}")


@app.cell
def _():
    print("## CI/CD Pipeline")
    print("")
    print("Typical ML CI/CD stages:")
    print("")
    print("```yaml")
    print("# .github/workflows/ml-pipeline.yml")
    print("on: push")
    print("jobs:")
    print("  test:")
    print("    runs-on: ubuntu-latest")
    print("    steps:")
    print("      - uses: actions/checkout@v4")
    print("      - run: make setup")
    print("      - run: make test")
    print("      - run: make train  # on main branch only")
    print("```")


@app.cell
def _():
    print("## Monitoring")
    print("")
    print("Production ML monitoring:")
    print("")
    print("1. **Performance metrics**")
    print("   - Prediction latency (p50, p99)")
    print("   - Throughput (predictions/second)")
    print("")
    print("2. **Model quality**")
    print("   - Prediction distribution over time")
    print("   - Feature drift detection")
    print("   - Actual vs predicted (when labels available)")
    print("")
    print("3. **Alerting**")
    print("   - RMSE exceeds threshold")
    print("   - Data drift detected")
    print("   - Prediction latency spike")


@app.cell
def _():
    print("## Retraining Triggers")
    print("")
    print("When to retrain:")
    print("  - Scheduled (weekly/monthly)")
    print("  - Performance degradation detected")
    print("  - Data drift detected")
    print("  - New feature available")
    print("  - Business requirements change")


@app.cell
def _():
    print("## Workshop Summary")
    print("")
    print("You've learned:")
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
