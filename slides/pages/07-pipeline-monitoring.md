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

</v-clicks>

<!-- Moved from "Go-live Questions" — the pain that motivates monitoring. -->

---

# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/pipeline_monitoring.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# What else? (Things we already know)

* Unit / Integration tests
* CI / CD
* Data quality
