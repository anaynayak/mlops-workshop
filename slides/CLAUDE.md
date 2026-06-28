# Slides — Claude Instructions

Slidev deck for the MLOps workshop. `slides.md` is the entry point; it holds only
the title slide and a list of `src:` includes. **All real content lives in
`pages/`** — one file per section, pulled in via:

```md
---
src: ./pages/06b-registry.md
---
```

To reorder/insert sections, edit the include list in `slides.md`. Page files can
contain multiple slides separated by `---`.

## Sections (dividers + breadcrumb)

Slidev has no native "section" concept; it's assembled from two pieces, both
driven off a custom `section:` frontmatter field:

1. **Divider slides** open each major section, inline in `slides.md`:
   ```md
   ---
   layout: section
   section: Model Registry
   ---
   # Model Registry
   One stable pointer to what's live.
   ```
2. **Persistent breadcrumb** (`global-bottom.vue`) shows the current section in
   the bottom-left of every content slide. It walks back to the nearest slide
   with a `section:` frontmatter, so you only tag each section's opening slide
   (the divider). It hides itself on `section`/`cover` layouts and when no
   section is set. `global-bottom.vue` is auto-loaded by Slidev from this dir.

To add a section: add a `layout: section` divider with a `section:` field in
`slides.md` before the section's `src:` includes. Nothing else needed — the
breadcrumb picks it up. The first non-divider section (Introduction) is tagged
via `section:` on its `src:` include block instead of a divider.

## Pedagogy: elastic tiers (read this before editing content)

The workshop teaches by **discover-by-building**: participants do a task manually,
feel the pain, then build the tool that solves it. Because pacing varies, every
section is authored in **tiers** the facilitator picks live:

- **Tier 0 — Talk (always covered):** diagram + pain-as-a-question + the answer.
  Even when out of time, everyone hears about every layer.
- **Tier 1 — Mini hands-on:** a short manual task that makes the pain felt + a tiny code step.
- **Tier 2 — Deep loop:** the full manual → pain → build arc.

**Rules when authoring slides:**
- Tag each slide's tier in a presenter note (`<!-- TIER 0 ... -->`) so the
  facilitator knows what is skippable vs mandatory.
- Hands-on blocks must be **detachable** — skipping one must never break the
  narrative of the surrounding Tier-0 slides.
- Don't pre-write the participant's notebook or the live code they're meant to
  discover. Slides show the *toolkit* (e.g. `mlops_workshop.registry` helpers) and
  the *concept*; participants wire the workflow themselves.

## Deep-dive sections & skip-to-review links

Each deep-dive (Tier 2) section has a **review/recap slide** with a `routeAlias`.
The section's entry slide carries a `<Link to="…">` to that alias. If the room
already self-discovered the topic, click the link to **jump straight to the recap,
skipping the Tier-1/2 walkthrough**, then continue.

Pattern:
```md
# Section Entry Slide
<div class="abs-br m-4 text-sm opacity-60">
<Link to="my-review">Skip to section recap →</Link>
</div>

... walkthrough slides ...

---
routeAlias: my-review
---
# Section — Recap
```

| Deep-dive section | File | Review alias | Status |
|---|---|---|---|
| Experimentation → MLflow | `pages/04-experiment.md` | `experiment-review` | done |
| Model Registry | `pages/06b-registry.md` | `registry-review` | done |

Tier-0 talk-only sections have **no** recap/skip-link (no walkthrough to skip):
- **Serving** (`pages/serving.md`) — batch→online transition; deployment strategies; bridges to Feature Stores via train-serve skew.
- **Monitoring** (`pages/07-pipeline-monitoring.md`) — layered monitoring, delayed labels.

When you build a new deep-dive section, add its recap slide + `routeAlias`, a skip
link on the entry slide, and a row to this table.

## Conventions

- Diagrams use the Excalidraw addon: `<Excalidraw drawFilePath="./draw/NAME.excalidraw" .../>`.
  Source files live in `draw/`. Don't reference a `drawFilePath` that doesn't exist
  (it breaks rendering) — use HTML/markdown layout if no diagram exists yet.
- Math renders via KaTeX (`$...$`, `$$...$$`).
- Layout uses UnoCSS utility classes in `<div class="...">` (grids, `abs-tr`, `mt-8`, etc.).
- Presenter notes go in `<!-- ... -->` at the end of a slide.

## Commands (run from `slides/`)

- `npm run dev` — live preview (`make slides` from repo root does the same)
- `npx slidev build --out /tmp/slidev-check` — validate the deck builds (catches
  bad includes, broken syntax). Run this after structural edits.
- `npm run export` — export to PDF.
