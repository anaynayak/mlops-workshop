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

<!--
- Context: Focus less on the ML problem statement. Would use it as a reference.
- Approach aspects that we build in layers.
-->


---
layout: image-right
image: /cab.png
backgroundSize: contain
---

We want to know how long a ride will take before the passenger gets in the car. Our dispatch system needs this to optimize pickup assignments and give riders accurate ETAs. 

— Product Team

<!--
- Domain: we're building a taxi fleet. Cars, drivers, riders, dispatch — the whole operation of getting people from A to B. This prediction is one small ask coming out of that business.
- Turn the product team's ask into a question for the audience: what data will we need to answer this?
-->

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

<!--
- Run these in background. Let me know if you run into issues.
- Will build some more context in the background.
- Open VSCode + Browser marimo session.
-->

---

# How the Model Works
<Excalidraw
  drawFilePath="./draw/model_primer.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# How Do We Measure "Good"?

Before we tune anything, we need a number that says how wrong the model is.

Take **4 taxi trips**. Compare what actually happened to what the model predicted:

| Trip | Actual (min) | Predicted (min) | Error | \|Error\| | Error² |
|---|---|---|---|---|---|
| A | 10 | 12 | +2 | 2 | 4 |
| B | 20 | 18 | −2 | 2 | 4 |
| C | 30 | 33 | +3 | 3 | 9 |
| D | 40 | 35 | −5 | 5 | 25 |

<div class="mt-4 text-sm opacity-70">

**Error = Predicted − Actual.** Positive = over-estimate, negative = under-estimate.
Every metric below is just a different way of summarising this one column of errors.

</div>

<!--
- The whole point: errors cancel out if you just average them (+2 −2 +3 −5 = −2). So we either take absolute value or square them.
-->

---

# MAE & MAPE — the "absolute error" family

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### MAE — Mean Absolute Error
Average of \|Error\|. **Same units as the target.**

$$\text{MAE} = \frac{2+2+3+5}{4} = 3 \text{ min}$$

> "On average we're off by **3 minutes**."

Treats all errors equally — a 10-min miss counts exactly 5× a 2-min miss. **Robust to outliers.**

</div>
<div>

### MAPE — Mean Absolute % Error
Each error as a **% of the actual** value. **Unit-free.**

$$\text{MAPE} = \frac{1}{4}\left(\tfrac{2}{10}+\tfrac{2}{20}+\tfrac{3}{30}+\tfrac{5}{40}\right) = 13.1\%$$

> "On average we're off by **13%**."

Great for communicating across scales — but **blows up for tiny actuals** (a 1-min trip).

</div>
</div>

---

# MSE & RMSE — the "squared error" family

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### MSE — Mean Squared Error
Average of Error². **Units are squared** (min²) — not human-readable.

$$\text{MSE} = \frac{4+4+9+25}{4} = 10.5 \text{ min}^2$$

Squaring **punishes big misses harder**: trip D's 5-min error contributes 25, not 5. This is the loss most models actually optimise.

</div>
<div>

### RMSE — Root Mean Squared Error
Square root of MSE → **back in minutes**.

$$\text{RMSE} = \sqrt{10.5} = 3.24 \text{ min}$$

Interpretable like MAE, but **outlier-sensitive**.

<div class="mt-4 text-sm opacity-80">

**RMSE vs MAE tells a story:** here 3.24 vs 3.0 — close, so errors are even. If RMSE were *much* bigger than MAE, a few large misses are hiding in your data.

</div>

</div>
</div>

---

# R² — how much better than just guessing the average?

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

The dumbest model always predicts the **mean** (here, 25 min). R² asks: how much of the variation does our model explain *over* that baseline?

$$R^2 = 1 - \frac{\text{model's squared error}}{\text{baseline's squared error}}$$

$$R^2 = 1 - \frac{42}{500} = \mathbf{0.916}$$

</div>
<div>

- **1.0** = perfect predictions
- **0.0** = no better than guessing the mean
- **< 0** = *worse* than guessing the mean

> "Our model explains **91.6%** of the variation in trip times."

**Unit-free**, so it's the easiest metric to compare across different problems.

</div>
</div>

<div class="text-sm opacity-70 mt-4">

Baseline error = (10−25)² + (20−25)² + (30−25)² + (40−25)² = 500. &nbsp; Model error = 4+4+9+25 = 42.

</div>

---

# Which Metric When?

<div class="grid grid-cols-2 gap-6 mt-6 text-sm">
<div>

| Metric | Units | Outliers | Reads as |
|---|---|---|---|
| **MAE** | minutes | shrugs | "off by 3 min" |
| **RMSE** | minutes | punishes | "off by 3.24 min" |
| **MSE** | min² | punishes | training loss |
| **MAPE** | % | n/a | "off by 13%" |
| **R²** | none | — | "explains 91.6%" |

</div>
<div>

- **Optimising / reporting error?** → **RMSE** (our workshop target). Same units as the goal, and it leans on big misses we care about.
- **Want a robust "typical miss"?** → **MAE**.
- **Talking to the business?** → **MAPE** ("13% off") or **R²** ("explains 91%").
- **MSE** is mostly the math under the hood — you rarely report it.

</div>
</div>

<div class="text-center mt-6 text-sm opacity-70">
All five describe the same errors — pick the lens that fits the audience.
</div>

---

# Where to Start

**Your goal: beat the baseline RMSE.**

Open `notebooks/02_train.py` and change one thing:

<div class="grid grid-cols-3 gap-6 mt-8">
<div>

### Features
Add or change columns the model sees

`is_weekend`, `traffic_hour`, `trip_distance_bucket`

</div>
<div>

### Model Type
Swap the algorithm

`XGBRegressor`, `GradientBoostingRegressor`, `LinearRegression`

</div>
<div>

### Hyperparameters
Tune model complexity

`n_estimators`, `max_depth`, `learning_rate`

</div>
</div>

<div class="text-center mt-8 text-sm opacity-70">
20 minutes — share your best RMSE
</div>

---

# Current state
<Excalidraw
  drawFilePath="./draw/current_state.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

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
  drawFilePath="./draw/pipeline_base.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/pipeline_training.excalidraw"
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
  drawFilePath="./draw/pipeline_inference.excalidraw"
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
  drawFilePath="./draw/pipeline_monitoring.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# Data Versioning

<v-clicks depth=2>

* MLFlow provides traceability 
* How do we get to reproducibility if the underlying data changes? 
  * History tables instead of in-place updates
  * Data version control systems e.g. https://dvc.org/  / https://git-lfs.com/
  * Tagged with experiments to provide full reproducibility.

</v-clicks>

---

# Feature Stores

<Excalidraw
  drawFilePath="./draw/feature_stores.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# What else? (Things we already know)

* Unit / Integration tests
* CI / CD
* Data quality

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
- **Books:** Designing Machine Learning Systems - Chip Huyen
- **Web** 
  - https://huyenchip.com/mlops/
  - https://ml-ops.org/

<div class="abs-br m-6 text-sm opacity-50">
  Thank you!
</div>
