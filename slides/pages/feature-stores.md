# One Feature, Two Consumers

<Excalidraw
  drawFilePath="./draw/pipeline_feature_store.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

<!--
TIER 0 — anchor the section in the architecture: the same route feature now feeds
both the offline training set and the online prediction path.
-->

---

# Add a Historical Route Feature

<div class="text-xl mt-6">

Make the model use <code>route_avg_duration_24h</code> in both places:

</div>

<div class="grid grid-cols-2 gap-8 mt-8">
<div>

### Training
Join the feature from trip history when you build the training set

</div>
<div>

### Serving
Fetch the latest value for the route before you predict

</div>
</div>

<div class="text-center mt-10 text-sm opacity-70">
5 minutes — sketch the wiring before you touch code.
</div>

<!--
TIER 1/2 ENTRY — let the room articulate why this feature is different from
trip_miles or pickup_hour. It exists in history and has to stay fresh online.
-->

---

# Why This Hurts Without a Feature Store

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

Without a feature store, you now own two jobs:

- compute the rolling route average for training
- reimplement the same definition for the API
- keep freshness and lookup keys aligned

</div>
<div>

<div class="text-2xl font-bold mt-8">
The feature is no longer just code in a notebook.
</div>

<div class="mt-8 text-lg">

It's history, a join key, a freshness policy, and a serving lookup.

</div>

</div>
</div>

<div class="text-center mt-8 text-sm opacity-70">
The pain is not storage. The pain is keeping training and serving consistent.
</div>

<!--
TIER 0 — name the actual problem. This is where train-serve skew becomes concrete.
-->

---

# Feast's Job in This Demo

<div class="grid grid-cols-3 gap-6 mt-8 text-sm">
<div>

### 1. Define
`feature_repo/` holds the entity, feature view, and feature service as code

</div>
<div>

### 2. Train
`get_historical_features(...)` gives a point-in-time-correct training set

</div>
<div>

### 3. Serve
`materialize(...)` loads the latest route values so the API can do a fast lookup

</div>
</div>

<div class="mt-10">

Toolkit, not magic:

```python
from mlops_workshop import feature_store

store, route_stats = feature_store.bootstrap_feature_store(raw_df)
training_df = feature_store.build_training_dataframe(raw_df, store=store)
online = feature_store.get_online_route_features(138, 236, store=store)
```

</div>

<!--
TIER 1/2 BUILD — show the helpers and the Feast moves, but keep the notebook/API
assembly discoverable for participants.
-->

---
routeAlias: feature-store-review
---

# Feature Stores — Recap

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

**The idea**

- one historical feature definition
- point-in-time retrieval for training
- latest-value lookup for serving
- less train-serve skew

</div>
<div>

**The moves**

```python
store, _ = feature_store.bootstrap_feature_store(raw_df)
training_df = feature_store.build_training_dataframe(raw_df, store=store)
online = feature_store.get_online_route_features(138, 236, store=store)
```

Train on `training_df`; serve with `online`.

</div>
</div>

<div class="text-center mt-8 text-sm opacity-70">
Landed here via the skip link? The room already discovered the pain — this is the consolidation.
</div>

<!--
TIER 0 — the short-form memory: define once, retrieve historically for training,
serve the latest value online.
-->
