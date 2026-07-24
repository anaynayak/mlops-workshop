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
routeAlias: setup
---

# Where We Start

The data scientist has done the exploratory work and shared a notebook with you.

**Get set up** — pick your platform (borrow from `snippets/`):

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

### macOS / Linux

<<< ../../snippets/setup_macos_linux.sh bash

</div>
<div>

### Windows (or no `make`)

<<< ../../snippets/setup_windows.sh bash

</div>
</div>

**Open the notebook:** `notebooks/00_setup.py` — Setup

<div class="text-sm opacity-70 mt-3">
Full Windows setup and gotchas: <code>docs/WINDOWS.md</code>.
Behind? <code>git checkout 02-registry</code> jumps to a stage's finished state.
</div>

The data: **20M NYC taxi trips** (FHVHV dataset) · Target: predict `trip_time` in seconds

<!--
- Run these in background. Let me know if you run into issues.
- This slide has routeAlias "setup" — the footer "Setup" link jumps back here anytime.
- Windows attendees: use `uv run poe <task>` (docs/WINDOWS.md), no make needed.
- Catch-up branches: 01-experimentation, 02-registry, 03-cicd, 04-feature-store.
- Open VSCode + Browser marimo session.
-->
