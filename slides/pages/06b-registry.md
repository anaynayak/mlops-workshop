# Ship Your Best Model

<div class="abs-br m-4 text-sm opacity-60">
<Link to="registry-review">Skip to section recap →</Link>
</div>

<div class="text-xl mt-6">

You've got a pile of runs in MLflow. The fleet can run **one** model.

</div>

<div class="text-2xl font-bold mt-8 text-center">
Make <code>04_inference.py</code> load <em>your best</em> model. Go.
</div>

<div class="text-center mt-10 text-sm opacity-70">
5 minutes — just get it working, however you can.
</div>

<!--
TIER 1/2 ENTRY — the manual pain. Don't help yet. Let them dig a run_id or
artifact path out of MLflow and paste it into the notebook. The fumbling IS
the lesson. If you're short on time, skip the hands-on and narrate the pain
on the next slide instead — the rest of the deck still works.
-->

---

# ...Now Do It Again

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

You hardcoded a path or run id:

```python
model = load_model("models/your_model.joblib")
# or: runs:/3f9c1a.../model
```

Then real life happens:

- "You trained a **better** one." → edit the path
- "That one's **worse** — go back." → ...which one was it?
- "Now do this for **everyone's** inference job." → edit them all

</div>
<div>

<div class="text-xl mt-10">

The model is **glued to the code**.

Every promotion is a code change.
Every rollback is a scramble.

</div>

<div class="mt-8 text-2xl font-bold">
We need a stable pointer to "the model that's live" —<br/>independent of code.
</div>

</div>
</div>

<!--
This is the felt pain. Tie it back to the Inference Questions slide:
"where do we manage joblib files? how do we roll back?" — you just lived it.
-->

---

# The Model Registry

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

A versioned home for models, **decoupled from code**.

```
nyc-taxi-duration
├── v1   ← @champion
├── v2
└── v3   ← @challenger
```

- Every promotion = a new **version**
- **Aliases** — `@champion` (live) and `@challenger` (candidate) — point at a version
- Consumers reference the **alias**, never a path or version number

</div>
<div>

<div class="mt-8">

**Promote** = move `@champion` to the challenger's version.

**Rollback** = move `@champion` back.

**Inference code never changes.**

</div>

<div class="mt-8 text-sm opacity-70">
MLflow gives us this for free — same tool we just used for tracking.
</div>

</div>
</div>

<!--
TIER 0 — always cover this slide, even if you skipped the hands-on. This is
the concept everyone must leave with: the model is an artifact, versioned and
addressed by alias, not a file glued into a script.
-->

---

# Promote With the Registry

Use the helpers in `mlops_workshop.registry` — you wire the steps together:

```python
from mlops_workshop import registry

# 1. Find a candidate — runs sorted best-first
registry.find_runs(metric="rmse")            # eyeball it, copy a run_id

# 2. Register that run's model as a new version
mv = registry.register_run("<run_id>")       # -> version N

# 3. Stage it as the challenger
registry.set_alias("challenger", mv.version)

# 4. It beats the champion? Promote it.
registry.set_alias("champion", mv.version)
```

Then the consumer — `04_inference.py` — stops caring about files:

```python
model = registry.load_model("champion")      # always the live model
```

<!--
TIER 1/2 BUILD. The notebook orchestration is THEIRS to write — these helpers
are just the toolkit. Granular on purpose: no one-call "promote_best()", so
they have to choose the winner and wire the flow themselves.
-->

---

# Rollback = Move the Alias

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

Ship a worse `v2`, then realise it in production:

```python
registry.list_versions()
#  version  aliases
#       2   [champion]   ← the bad one
#       1   []

registry.set_alias("champion", 1)
```

</div>
<div>

<div class="mt-6 text-xl">

`04_inference.py` is **untouched**. It still says:

```python
registry.load_model("champion")
```

…it just resolves to v1 again.

</div>

</div>
</div>

<div class="text-center mt-8 text-lg font-bold">
The consumer and the decision of "what's live" are finally separate.
</div>

---
routeAlias: registry-review
---

# Registry — Recap

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

**The idea**

- A model is a **versioned artifact**, not a file glued into code
- `@champion` points at the live version; `@challenger` is the candidate
- Inference loads the **alias** — so promotion and rollback never touch consumer code

</div>
<div>

**The moves** — `mlops_workshop.registry`

```python
registry.find_runs(metric="rmse")             # pick a candidate
mv = registry.register_run(run_id)            # -> new version
registry.set_alias("challenger", mv.version)  # stage it
registry.set_alias("champion", mv.version)    # promote when it wins
registry.load_model("champion")               # consumer loads
```

Rollback = `set_alias("champion", <older version>)`.

</div>
</div>

<div class="text-center mt-8 text-sm opacity-70">
Landed here via the skip link? You discovered it yourselves — this is the consolidation.
</div>

---

# On an Ongoing Basis

<div class="mt-6">

What we did by hand, you'd eventually automate:

</div>

<div class="grid grid-cols-3 gap-6 mt-8 text-sm">
<div>

### Today (manual)
Eyeball runs → register → move the alias yourself

</div>
<div>

### Next
A job that auto-registers the best run and promotes it on a schedule

</div>
<div>

### Production
CI/CD gates: validate against thresholds before the alias moves

</div>
</div>

<div class="text-center mt-10 text-sm opacity-70">
Same mechanism — the alias is still the single source of "what's live."
</div>

<!--
TIER 0 — where this goes next. We deliberately do NOT build the automation; it
drags in scheduling and CI/CD. Name it, draw the arrow, move on.
-->

---

# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/pipeline_registry.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# Picking the Champion

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

A challenger isn't the champion just because its RMSE looks good.

- The champion was trained on **last quarter's** data
- The challenger trained on **new** data — different trips, different conditions
- A "perfect" champion can look unbeatable on the old yardstick — or a challenger can win for the **wrong reasons**

</div>
<div>

<div class="text-xl mt-8">

Comparing raw metrics is **not enough**.

</div>

<div class="mt-6">

We need a deliberate **validation** step before promoting:

- Evaluate both on a **fair, current** holdout
- Guardrails / thresholds the challenger must clear
- Then decide — does it *actually* beat the champion?

</div>

</div>
</div>

<!--
The conversation that motivates Model Validation. The trap: judging challenger vs
champion on mismatched data. Ask the room — "your champion was perfect last
quarter; with new data, can a challenger ever win, and how would you know it's
real?" The validation stage on the next diagram is the answer.
-->

---

# The MLOps Pipeline
<Excalidraw
  drawFilePath="./draw/pipeline_validation.excalidraw"
  class="w-[800px]"
  :darkMode="false"
  :background="false"
/>

---

# Questions the Registry Should Answer

<v-clicks depth=2>

* Where do we manage model files in production — and trace them back to source?
* How do we ensure the next promoted model is better than the last?
  * Accuracy metrics
  * Challenger vs champion — and should we compare them?
* How do we roll back to a previous version?

</v-clicks>

<!-- Moved from "Inference Questions". Review coverage against this section. -->
