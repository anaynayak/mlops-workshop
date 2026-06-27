# Where to Start

<div class="abs-br m-4 text-sm opacity-60">
<Link to="experiment-review">Skip to section recap →</Link>
</div>

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

# Now Multiply That by 50

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

Everyone in this room just tried a different idea:

- "I added `is_weekend`." → 6.1 min
- "I switched to XGBoost." → 5.8 min
- "depth=15, then 20, then..." → 5.9, 6.0...

Now it's **one person, next week, 50 runs deep.**

</div>
<div>

<div class="text-xl mt-8">

- Which change gave that **5.8**?
- What were the exact params?
- Can you **reproduce** it tomorrow?
- Whose notebook had the good one?

</div>

<div class="mt-8 text-2xl font-bold">
Scattered in notebooks and memory.<br/>This doesn't scale.
</div>

</div>
</div>

<!--
This is the felt pain — and it lands hardest if the room genuinely tried
different things in the exercise above. Collect a few of their RMSEs out loud
first, then ask: "which knob got you there, exactly?" Watch them not remember.
-->

---

# MLflow Tracking

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

A central store for every run — no more scattered notebooks.

```
experiment: nyc-taxi-duration
├── run: rf_shallow    rmse=419  depth=5
├── run: rf_baseline   rmse=386  depth=10
└── run: rf_deeper     rmse=368  depth=15
```

Each **run** captures:

- **Params** — what you changed
- **Metrics** — what you got
- **Artifacts** — the model itself

</div>
<div>

<div class="mt-8">

Compare, sort, and reproduce — from one place.

</div>

<div class="mt-8 text-sm opacity-70">
We only <strong>track</strong> here. Choosing what to ship comes later
(the Model Registry).
</div>

</div>
</div>

<!--
TIER 0 — always cover. The mental model everyone must leave with: an experiment
holds many runs; each run = params + metrics + artifacts, in one comparable store.
-->

---

# Track Your Runs

Wrap each run — you add this to your training loop:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("nyc-taxi-duration")

with mlflow.start_run(run_name="rf_baseline"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    model.fit(X_train, y_train)
    metrics = evaluate_model(y_test, model.predict(X_test))

    mlflow.log_metric("rmse", metrics["rmse"])
    mlflow.log_metric("r2", metrics["r2"])
    mlflow.sklearn.log_model(model, name="model")   # artifact only — register later
```

Then explore them: `make mlflow`

<!--
TIER 1/2 BUILD. Participants add this to their own run from the exercise. We log
the model as an ARTIFACT only — no registered_model_name here. Registration is a
separate, deliberate step in the registry section, so each loop has its own
discovery moment.
-->

---
routeAlias: experiment-review
---

# Experiment Tracking — Recap

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

**The idea**

- 50 scattered runs → **one comparable store**
- An **experiment** holds many **runs**
- Each run = **params + metrics + artifact**
- Sort, compare, reproduce — and it sets up promotion later

</div>
<div>

**The moves**

```python
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("nyc-taxi-duration")

with mlflow.start_run(run_name="..."):
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, name="model")
```

Browse it all with `make mlflow`.

</div>
</div>

<div class="text-center mt-8 text-sm opacity-70">
Landed here via the skip link? You lived the pain already — this is the consolidation.
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
