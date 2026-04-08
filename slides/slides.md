---
theme: default
title: MLOps Workshop
---

# MLOps Workshop

From Notebook to Production in 2 Hours

<div class="abs-br m-6 text-sm opacity-50">
  NYC Taxi Trip Duration Prediction
</div>

---

# The Problem

> "I want to know how long a ride will take before the passenger gets in the car. Our dispatch system needs this to optimize pickup assignments and give riders accurate ETAs. The data science team has something working — make it production-ready."

— CEO

---

# Where We Start

The data scientist has done the exploratory work and shared a notebook with you.

**Get set up:**

```bash
git clone <repo-url>
make setup
make data
make sample
make lab
```

**Open the notebook:**
- `notebooks/01_explore_data.py` — EDA with visualizations

The data: **20M NYC taxi trips** (FHVHV dataset)
Target: predict `trip_time` in seconds

---

# The Question

<div class="text-center mt-20">
  <div class="text-6xl font-bold mb-8">
    What does it take<br/>to run this in production?
  </div>
  <div class="text-xl opacity-70">
    That's what we'll answer in the next 2 hours.
  </div>
</div>

---

# Workshop Stages

| Stage | Topic | Notebook |
|-------|-------|----------|
| 0 | Setup & Environment | `00_setup.py` |
| 1 | Data Exploration | `01_explore_data.py` |
| 2 | Baseline Model | `02_baseline.py` |
| 3 | Tuning + MLflow | `03_tuning.py` |
| 4 | Batch Inference | `04_inference.py` |
| 5 | Operations | `05_operations.py` |

---

# Stage 0-1: Understand the Data

Key findings from EDA:

- **2M trips** in our sample (10% of 20M)
- Mean trip: **18.8 minutes**, Median: **15.3 minutes**
- Strong predictor: `trip_miles` (correlation: 0.78)
- 262 unique pickup/dropoff locations
- Clear patterns by hour and day of week

Features selected: `trip_miles`, `PULocationID`, `DOLocationID`, `pickup_hour`, `day_of_week`

---

# Stage 2: Baseline Model

Random Forest — 100 trees, max depth 10

```
RMSE: 6.44 minutes
R²:   0.7543
MAE:  4.32 minutes
```

**What this means:**
- Explains **75%** of trip duration variance
- Typical prediction off by **~4 minutes**
- Decent for 5 features, no tuning

**Top features:** `trip_miles` > `PULocationID` > `DOLocationID`

---

# Stage 3: Experiment Tracking

Why track experiments?

- **Reproducibility** — which parameters produced which results?
- **Comparison** — is this model better than the last one?
- **Auditability** — who trained what, when?

**MLflow tracks:**
- Parameters (n_estimators, max_depth)
- Metrics (RMSE, R²)
- Models (versioned artifacts)

```bash
make train   # trains and logs to MLflow
make mlflow  # opens the UI
```

---

# Stage 4: Batch Inference

Production pattern for ride duration prediction:

1. Load the trained model
2. Score new trip data in batches
3. Write predictions to storage

```python
model = load_model("models/rf_model.joblib")
predictions = model.predict(new_data)
```

**Model Registry concepts:**
- Register → Stage → Validate → Production
- Version models for rollback capability

---

# Stage 5: Operations

What makes ML production-ready?

**Testing**
- Unit tests for features/transforms
- Model validation (RMSE < threshold)
- Data quality checks

**CI/CD**
- Automated test pipeline
- Retrain on schedule or drift detection

**Monitoring**
- Prediction distribution over time
- Feature drift detection
- Performance degradation alerts

---

# Recap

<div class="grid grid-cols-2 gap-8 mt-8">
<div>

### What we built
- EDA pipeline with visualizations
- Feature engineering module
- Trained Random Forest model
- MLflow experiment tracking
- Batch inference pipeline

</div>
<div>

### What we learned
- ML starts messy — MLOps makes it reliable
- Track experiments from day one
- Test data, models, and pipelines
- Monitor for drift and degradation
- Automate everything you can

</div>
</div>

---

# Resources

- **Workshop repo:** `github.com/anaynayak/mlops-workshop`
- **MLflow docs:** mlflow.org
- **Slidev docs:** sli.dev
- **Marimo docs:** marimo.io

<div class="abs-br m-6 text-sm opacity-50">
  Thank you!
</div>
