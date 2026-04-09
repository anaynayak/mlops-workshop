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
- `notebooks/00_setup.py` — Setup

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
<Excalidraw
  drawFilePath="./draw/02.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# Training Questions

<v-clicks depth=2>

* How do we know which model is the best? 
  * 10s / 100s of models across hyperparams / models
* How can we trace / recreate the same model ? 
  * Different features / code
  * Hyper params / models
  * Artifact lineage to source
* Where does this cycle run? Which environments? 
  * In Dev / Staging / Prod
* What gets promoted to the next environment?
  * Is it the code?
  * Is it the model? What are the trade-offs?

</v-clicks>

---

# Inference Questions

<v-clicks depth=2>

* Where do we manage the joblib files in production?
  * How do we trace it back to the source artifacts?
* How do we ensure that the next model promoted is better than previous?
  * Accuracy metrics
  * Challenger v/s champion
  * Should we compare challenger v/s champion metrics?
* How do we roll-back to previous model version?

</v-clicks>

---

# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/03.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# Go-live Questions

<v-clicks depth=2>

* What can go wrong?
  * Schema changes
  * Missing data 
  * feature drift e.g. seasonal variations (holiday season, school start)
* Can the model degrade in production?
  * Inference drift
  * Concept drift
* How do we safeguard against such issues?

</v-clicks>

---

# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/04.excalidraw"
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

# Resources

- **Workshop repo:** `github.com/anaynayak/mlops-workshop`
- **MLflow docs:** mlflow.org
- **Slidev docs:** sli.dev
- **Marimo docs:** marimo.io
- **Books:** Designing Machine Learning Systems - Chip Huyen
- **Web** 
  - https://huyenchip.com/mlops/
  - https://ml-ops.org/

<div class="abs-br m-6 text-sm opacity-50">
  Thank you!
</div>
