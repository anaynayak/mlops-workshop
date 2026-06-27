# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/pipeline_base.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/pipeline_training.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# Training Questions

<v-clicks depth=2>

* How do we know which model is the best? 
  * 10s / 100s of models across hyperparams / models
* How can we trace / recreate the same model ? 
  * Different features / code
  * Hyper params / models
  * Artifact lineage to source
* Where does this cycle run? Which environments? 
  * In Dev / Staging / Prod
* What gets promoted to the next environment?
  * Is it the code?
  * Is it the model? What are the trade-offs?

</v-clicks>
