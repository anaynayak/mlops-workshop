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
