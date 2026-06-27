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
