# MLOps Workshop

From Notebook to Production in 2 Hours.

A hands-on workshop for platform engineers, data engineers, and data scientists new to MLOps. You'll take a data science notebook and learn what it takes to run it in production.

## Problem

Predict NYC taxi trip duration (in seconds) using the FHVHV dataset (~20M trips). The dispatch system needs accurate ETAs to optimize pickup assignments.

## Quick Start

```bash
git clone https://github.com/anaynayak/mlops-workshop
make setup
make data
make sample
make lab
```

## Notebooks

| Notebook | Topic |
|---|---|
| `00_setup.py` | Environment setup and data loading |
| `01_explore_data.py` | Exploratory data analysis |
| `02_baseline.py` | Baseline model training |
| `03_tuning.py` | Hyperparameter tuning with MLflow tracking |
| `04_inference.py` | Batch inference and model serving |

## Commands

| Command | Description |
|---|---|
| `make setup` | Install dependencies with uv |
| `make data` | Download dataset (requires `WORKSHOP_SAMPLE_URL`) |
| `make sample` | Create a sample from the full dataset |
| `make lab` | Launch marimo notebooks |
| `make train` | Run training script |
| `make infer` | Run inference script |
| `make test` | Run tests |
| `make mlflow` | Launch MLflow UI |
| `make slides` | Launch slide deck |

## What You'll Learn

The workshop covers the full MLOps pipeline:

- **Feature Engineering** — Transform raw data into ML features (`trip_miles`, `pickup_hour`, `PULocationID`)
- **Experimentation** — Track every training run with MLflow (parameters, metrics, artifacts)
- **Model Training** — Fit and compare models (Random Forest, XGBoost, etc.)
- **Validation** — Test model performance against thresholds (RMSE, R², MAE)
- **Promotion** — Register and stage models in a Model Registry (v1, v2, v3...)
- **Inference** — Score new data in production (batch or real-time)
- **Monitoring** — Detect feature drift and inference drift, alert on degradation
- **Data Versioning** — Reproducibility through history tables and data version control

## Resources

- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen
- [mlops.org](https://ml-ops.org/)
- [huyenchip.com/mlops](https://huyenchip.com/mlops/)
