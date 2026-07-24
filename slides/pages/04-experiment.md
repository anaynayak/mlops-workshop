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

Borrow this from `snippets/tracking.py` — don't retype it:

<<< ../../snippets/tracking.py python

<div class="mt-2 text-sm opacity-70">
<strong>marimo gotcha:</strong> each variable lives in exactly one cell. This one cell
defines <code>model</code> and <code>metrics</code>, so it <em>replaces</em> the separate
train and evaluate cells — don't redeclare them. Then explore runs: <code>make mlflow</code>
</div>

<!--
TIER 1/2 BUILD. Participants borrow snippets/tracking.py into 02_train.py. The
marimo teaching point: a name is defined once, so the tracking cell must REPLACE
the separate train + evaluate cells and return BOTH model and metrics — that's why
we combined them. We log the model as an ARTIFACT only (no registered_model_name);
registration is a separate, deliberate step in the registry section so each loop
keeps its own discovery moment.
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
  drawFilePath="./draw/pipeline_experiments.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

<!--
Where we are now: features → training → experiments pushed to the tracking store.
Inference is faded out — we haven't taken this to production yet. That gap is the
hook: everything offline works, but nothing is actually serving. It sets up the
question that drives the rest of the workshop — how do we take this to prod?
-->

---

# Questions Experimentation Raises

<v-clicks depth=2>

* How do we know which model is the best?
  * 10s / 100s of models across hyperparameters / model types
* How do we trace and recreate the exact same model?
  * Which features / code
  * Which hyperparameters / model
  * Artifact lineage back to source

</v-clicks>

<!--
Moved from the old "Training Questions". Review: does the Experimentation /
MLflow tracking section actually answer these?
-->

