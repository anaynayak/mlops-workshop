# The MLOps Pipeline — the Map

<div class="text-sm opacity-70 -mt-2 mb-2">
You understand the model and how we score it. Here's everything we build from here:
</div>

<Excalidraw
  drawFilePath="./draw/pipeline_overview.excalidraw"
  class="w-[820px]"
  :darkMode="false"
  :background="false"
/>

<!--
TIER 0 — the roadmap. Introduce the whole map right after the model + metrics
concepts: raw data → features → training → experiments → validation → promotion
(via the registry) → inference. Everyone now has a mental model to hang each
section on; the per-stage pipeline diagrams later light up one piece of THIS map.
Keep it to ~60 seconds: name the stages, promise we build each, move on.
-->
