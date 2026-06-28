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
| `01_features.py` | Feature engineering |
| `02_train.py` | Model training |
| `03_experiment.py` | Experimentation with MLflow tracking |
| `04_inference.py` | Batch inference |
| `05_feature_store.py` | Feast feature store and serving endpoint |

## Commands

| Command | Description |
|---|---|
| `make setup` | Install dependencies with uv |
| `make data` | Download the workshop sample (override with `WORKSHOP_SAMPLE_URL`) |
| `make sample` | Create a sample from the full dataset |
| `make lab` | Launch marimo notebooks |
| `make train` | Run training script |
| `make infer` | Run inference script |
| `make feature-store` | Build Feast data, materialize online features, and train the feature-store model |
| `make serve` | Launch the prediction API backed by Feast online features |
| `make test` | Run tests |
| `make mlflow` | Launch MLflow UI |
| `make slides` | Launch slide deck |

## Slides

The Slidev deck lives in `slides/`.

- Local preview: `make slides`
- Static build: `cd slides && npm run build`
- GitHub Pages: pushes to `main` automatically build and deploy the deck to Pages

## What You'll Learn

The workshop covers the full MLOps pipeline:

- **Feature Engineering** — Transform raw data into ML features (`trip_miles`, `pickup_hour`, `PULocationID`)
- **Experimentation** — Track every training run with MLflow (parameters, metrics, artifacts)
- **Model Training** — Fit and compare models (Random Forest, XGBoost, etc.)
- **Validation** — Test model performance against thresholds (RMSE, R², MAE)
- **Promotion** — Register and stage models in a Model Registry (v1, v2, v3...)
- **Inference** — Score new data in production (batch or real-time)
- **Feature Stores** — Reuse the same historical route feature for training and a live endpoint
- **Monitoring** — Detect feature drift and inference drift, alert on degradation
- **Data Versioning** — Reproducibility through history tables and data version control

## Feature Store Demo

Build the Feast repo, materialize online features, and train the model:

```bash
make feature-store
```

Then start the serving endpoint:

```bash
make serve
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_miles": 4.2,
    "PULocationID": 138,
    "DOLocationID": 236,
    "pickup_datetime": "2024-01-08T09:15:00"
  }'
```

## Resources

- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen
- [mlops.org](https://ml-ops.org/)
- [huyenchip.com/mlops](https://huyenchip.com/mlops/)
