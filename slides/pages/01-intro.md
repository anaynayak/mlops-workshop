---
layout: image-right
image: /cab.png
backgroundSize: contain
---

We want to know how long a ride will take before the passenger gets in the car. Our dispatch system needs this to optimize pickup assignments and give riders accurate ETAs. 

— Product Team

<!--
- Domain: we're building a taxi fleet. Cars, drivers, riders, dispatch — the whole operation of getting people from A to B. This prediction is one small ask coming out of that business.
- Turn the product team's ask into a question for the audience: what data will we need to answer this?
-->

---

# Where We Start

The data scientist has done the exploratory work and shared a notebook with you.

**Get set up:**

```bash
git clone https://github.com/anaynayak/mlops-workshop
make setup
make data
make sample
make lab
```

**Open the notebook:**
- `notebooks/00_setup.py` — Setup

The data: **20M NYC taxi trips** (FHVHV dataset)
Target: predict `trip_time` in seconds

<!--
- Run these in background. Let me know if you run into issues.
- Will build some more context in the background.
- Open VSCode + Browser marimo session.
-->
