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
