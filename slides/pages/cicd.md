# CI/CD Pipeline

<Excalidraw
  drawFilePath="./draw/pipeline_cicd.excalidraw"
  class="w-[820px]"
  :darkMode="false"
  :background="false"
/>

<!--
The ML-specific twist, not generic CI: CODE and MODEL are two artifacts with two
promotion paths. Code rides the usual gates (lint/test/security); the model rides
the registry (challenger → champion) gated by validation. Pre-prod often can't see
prod data, so the real model is trained & promoted in prod — pre-prod only proves
the pipeline runs. And rollback is two independent levers: redeploy code vs move
the @champion alias.
-->

---

# Questions to Address

<v-clicks depth=2>

* Where does this cycle run? Which environments?
  * Dev / Pre-prod / Prod
* What gets promoted to the next environment?
  * The code? The model? What are the trade-offs?
* What must pass before code ships?
  * Tests, lint, security / dependency scan, review
* Train once and promote the model artifact, or retrain per environment — and why?
  * Data availability, governance
* If prod data isn't available in pre-prod, where is the model actually trained and validated?
* What triggers a retrain?
  * Schedule, drift, manual — a code merge isn't a retrain
* How is a deployed model traced back to its code SHA + data version?

</v-clicks>

<!--
Moved/expanded from the old "Training Questions". The last point bridges to the
Data Versioning section.
-->
