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
