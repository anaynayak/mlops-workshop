# It's All Been Batch

<div class="text-xl mt-6">

Step back at everything we've built. It scores data **in bulk, on a schedule**:

</div>

- `04_inference.py` reads a table of trips → predicts → writes the results
- Monitoring watches those batches roll by

That's a real ML system — plenty of them ship exactly like this.

<div class="grid grid-cols-2 gap-8 mt-8">
<div>

But remember the ask:

> an ETA **before the passenger gets in the car**

A rider is at the curb **now**. A job that ran last night can't answer them.

</div>
<div>

<div class="text-2xl font-bold mt-2">
Real-time isn't "batch, but faster."<br/>It's a different system.
</div>

</div>
</div>

<!--
TIER 0 — the transition slide. Everything up to here has quietly been batch:
the inference notebook scores a table, monitoring watches batches. Name that out
loud, then snap back to the original product ask to motivate crossing into online
serving. This whole section is talk — there's no clean hands-on for standing up a
service in the room.
-->

---

# Two Ways to Run the Same Model

<div class="grid grid-cols-2 gap-8 mt-6 text-sm">
<div>

### Batch
Score many rows on a schedule

- Reads / writes a table
- High throughput, cheap, simple
- *Nightly demand forecast per zone*

❌ Stale the moment a rider asks

</div>
<div>

### Online / real-time
One request → one prediction, synchronously

- Behind an API, **latency budget in ms**
- Always-on, scales with traffic
- *ETA at dispatch* — **our case**

</div>
</div>

<div class="text-center mt-10 text-lg">
Same <code>@champion</code> model — the choice is <em>how and when</em> it runs.
</div>

<!--
TIER 0. The point: serving mode is a design decision driven by the use case, not
a default. Our latency requirement forces online. Mention batch is still the
right answer for many ML systems — don't oversell real-time.
-->

---

# The Script Becomes a Service

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

The batch job turns into a long-lived service:

```python
# on startup — same alias, still the source of truth
model = registry.load_model("champion")

@app.post("/predict")
def predict(trip):
    return model.predict(features(trip))
```

The registry alias still decides what's live. Now you also deploy the **service** around it.

</div>
<div>

<div class="mt-2">

What's new once it's online:

- **Latency** — the request is waiting; ms matter. Those operational signals from monitoring become **SLAs**.
- **Always-on & scaling** — survive rush hour, not a nightly run
- **Packaging** — ship model + deps as a container, runs anywhere

</div>

</div>
</div>

<!--
TIER 0. Tie back: the operational layer (latency/errors/throughput) we named in
monitoring is exactly what you now have to honour as a contract. The model didn't
change — the system around it did.
-->

---

# Where Do the Features Come From?

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

<Excalidraw
  drawFilePath="./draw/serving.excalidraw"
  class="w-[460px]"
  :darkMode="false"
  :background="false"
/>

</div>
<div>

The request is a **raw trip** — pickup, dropoff, timestamp. The model needs `pickup_hour`, `trip_miles`, `PULocationID`…

- In **training**, features were computed in **batch**, from the table.
- **Online**, you compute them **live, per request**, in milliseconds.

<div class="mt-4 text-lg font-bold">
One feature, defined twice, on two code paths.
</div>

If they drift apart, the model sees inputs it never trained on — **train-serve skew**.

<div class="mt-4 text-sm opacity-70">
Sound like something that needs an owner? That's the feature store →
</div>

</div>
</div>

<!--
TIER 0 — the bridge to Feature Stores. Going online is what makes the feature
store concrete: batch precomputed features in a table vs recomputing the same
logic live. Plant "train-serve skew" here; the Feature Stores section pays it
off. Don't solve it now — just make them feel the duplication.
-->

---

# Putting a Challenger in Front of Real Traffic

<div class="text-sm opacity-70 mt-2">In batch you just rerun with the new model. Online, every change hits real riders. How does <code>@challenger</code> earn <code>@champion</code> — without betting the fleet?</div>

<div class="mt-6 space-y-2 text-sm">

- **Big-bang / recreate** — swap to v2 everywhere at once.  ·  *simplest; riskiest — a bad model hits 100% instantly*
- **Blue-green** — full v2 stack beside v1, flip all traffic over.  ·  *instant rollback (flip back), double the infra*
- **Canary** — champion serves most traffic, challenger a slice (~5%); widen if healthy.  ·  *small blast radius*
- **Shadow / mirror** — challenger sees real requests, predictions **logged, never served**.  ·  *zero rider risk; validate on live traffic before promoting*
- **A/B test** — split traffic, compare a **business** metric.  ·  *did pickups get faster — not just lower RMSE?*

</div>

<div class="text-center mt-8 text-lg font-bold">
These are how a challenger earns champion. Rollback is still: move the alias / flip back.
</div>

<!--
TIER 0 — the slide that motivated this whole section. Map each strategy onto the
champion/challenger language they already own. Shadow is the "aha": the challenger
gets production inputs but its outputs never reach a rider, so you compare it to
the live champion at zero risk. Canary and shadow lean on the monitoring layers
from the previous section to decide "healthy."
-->

---

# Questions Serving Should Answer

<v-clicks depth=2>

* Batch or online — what does this use case actually need?
  * Latency budget? How fresh must the answer be?
* Where does the service run, and how does it scale with peak traffic?
* How do we roll out a new champion without risking the fleet?
  * Canary / shadow / blue-green — which fits, and why?
* Is the new version actually better *in production*?
  * A technical metric (RMSE) vs a business metric (pickup time)

</v-clicks>

<!--
The deployment-strategy tradeoff shows up both on its own slide and here as a
question, so it lands whether or not you walk the previous slide in full.
-->
