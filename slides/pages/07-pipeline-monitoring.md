# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/pipeline_monitoring.excalidraw"
  class="w-[820px]"
  :darkMode="false"
  :background="false"
/>

<!--
The payoff: the pipeline becomes a cycle. Drift on the inputs (Feature Drift) or
the outputs/quality (Inference Drift) feeds back into retraining — challenger →
validate → promote. Monitoring is what closes the loop back to experimentation.
-->

---

# You Can't See Accuracy Yet

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

The model predicts an ETA **now**.

The **actual** `trip_time` only lands when the trip ends — and only
counts once it's joined back to the prediction. Hours, sometimes days, later.

So your true error (RMSE) is always **looking at the past**.

</div>
<div>

<div class="text-xl mt-6">
By the time accuracy <em>looks</em> bad, you've already served bad ETAs for days.
</div>

<div class="mt-8 text-2xl font-bold">
You need leading indicators —<br/>signals that fire <em>before</em> the labels arrive.
</div>

</div>
</div>

<!--
The trap beginners miss: "we'll just watch RMSE in prod." You can't — ground
truth is delayed. Capturing actuals and joining them back is itself an
engineering task (ties to Data Versioning). So we watch proxies first.
-->

---

# Monitor in Layers

<div class="text-sm opacity-70 mt-2">Cheap & early → expensive & late. Each layer warns before the one below it.</div>

<div class="mt-6 space-y-2">
<v-clicks>

- **Operational** — latency, errors, throughput  ·  *is the service even up?*
- **Data quality** — schema, nulls, ranges, missing features  ·  *a new `PULocationID` appears*
- **Feature drift** — input distribution shifts vs training  ·  *post-event traffic patterns*
- **Prediction drift** — output distribution shifts  ·  *predicted ETAs creep up*
- **Model quality** — accuracy on the actuals that finally landed  ·  *concept drift: a new road*

</v-clicks>
</div>

<div class="grid grid-cols-2 gap-8 mt-8 text-sm">
<div class="opacity-80">

**Top 4: no ground truth needed** — detectable immediately.

</div>
<div class="opacity-80">

**Bottom: needs labels** — the slowest, truest signal.

</div>
</div>

<!--
The argument: you don't wait for accuracy to tank. The cheap proxies (drift on
inputs/outputs) fire first. Caveat to say aloud: drift ≠ degradation — don't
auto-retrain on every drift alert (cost + instability).
-->

---

# What Could Go Wrong?

<v-clicks depth=2>

* What can go wrong?
  * Schema changes
  * Missing data
  * Feature drift — e.g. seasonal variations (holiday season, school start)
* Can the model degrade in production?
  * Inference drift
  * Concept drift
* How do we safeguard against these?
  * Alert thresholds, automated retrain triggers, rollback to the previous champion

</v-clicks>
