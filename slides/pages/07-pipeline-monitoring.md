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
