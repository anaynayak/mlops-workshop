# Snippets — borrow, don't retype

These are the code blocks the slides show, kept here as **runnable files** so you
can copy them straight into your notebook instead of squinting at a projector.
Each slide renders the file directly (via Slidev's `<<<` include), so what you see
on screen and what you copy here are always the same thing.

| File | Where it's used | What it does |
|---|---|---|
| `tracking.py` | Experimentation | The MLflow tracking cell for `notebooks/02_train.py`. marimo note: this one cell defines `model` and `metrics`, so it *replaces* the separate train + evaluate cells (a variable is defined in only one cell). |
| `registry_promote.py` | Model Registry | Find a run, register it (from the **API**, not the UI), stage `@challenger`, promote `@champion`, load by alias. |
| `registry_rollback.py` | Model Registry | Move `@champion` back to an older version — rollback without touching consumer code. |
| `serving_app.py` | Serving | The batch script turned into an online `/predict` service that loads `@champion`. |
| `feast_demo.py` | Feature Stores | The Feast building blocks (historical + online feature retrieval). |
| `setup_macos_linux.sh` | Setup | First-run commands for macOS/Linux (`make`). |
| `setup_windows.sh` | Setup | First-run commands for Windows or anywhere without `make` (`uv run poe`). |

Tracking store: everything defaults to the local `sqlite:///mlruns/mlflow.db`. To run
against a hosted MLflow instead, set `MLFLOW_TRACKING_URI` — no code change. See
[`../docs/CLOUD.md`](../docs/CLOUD.md).
