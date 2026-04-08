---
theme: default
title: MLOps Workshop
drawings:
  enabled: false
contextMenu: false
---

# MLOps Workshop

From Notebook to Production in 2 Hours

<div class="abs-br m-6 text-sm opacity-50">
  NYC Taxi Trip Duration Prediction
</div>

---
layout: quote
---

We want to know how long a ride will take before the passenger gets in the car. Our dispatch system needs this to optimize pickup assignments and give riders accurate ETAs. 

— Product Team

---

# Where We Start

The data scientist has done the exploratory work and shared a notebook with you.

**Get set up:**

```bash
git clone https://github.com/anaynayak/mlops-workshop
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

# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/01.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# The MLOps Pipeline

<div class="grid grid-cols-5 gap-4 text-center text-sm mt-4">
<div>

### Feature Engineering
Transform raw data into ML features

`trip_miles`, `pickup_hour`, `PULocationID`...

</div>
<div>

### Model Training
Fit model on training data

Random Forest, XGBoost, etc.

</div>
<div>

### Validation
Test model performance

RMSE, R², MAE thresholds

</div>
<div>

### Promotion
Register & stage model

Model Registry v1, v2, v3...

</div>
<div>

### Inference
Score new data in production

Batch or real-time

</div>
</div>

<div class="grid grid-cols-4 gap-4 text-center text-sm mt-8">
<div>

### Experimentation
Track every training run

Parameters, metrics, artifacts

</div>
<div>

### Model Registry
Version and stage models

Staging → Production

</div>
<div>

### Feature Drift
Monitor input distributions

Retrain when data shifts

</div>
<div>

### Inference Drift
Monitor prediction quality

Alert on degradation

</div>
</div>

---

# Workshop Stages

| Stage | Topic | Pipeline Step | Notebook |
|-------|-------|---------------|----------|
| 0 | Setup & Environment | — | `00_setup.py` |
| 1 | Data Exploration | Feature Engineering | `01_explore_data.py` |
| 2 | Baseline Model | Training + Validation | `02_baseline.py` |
| 3 | Tuning + MLflow | Experimentation | `03_tuning.py` |
| 4 | Batch Inference | Inference + Registry | `04_inference.py` |
| 5 | Operations | Drift + Monitoring | `05_operations.py` |

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
