# Running in the cloud (escape hatch)

The workshop is designed to run **locally** — that's the highest-fidelity way to
learn MLOps, and it works on macOS, Linux, and Windows (see
[`WINDOWS.md`](./WINDOWS.md)). This page is a fallback for the rare machine that
won't cooperate and would otherwise eat workshop time.

The two problems a cloud run has to solve are **compute** (run the notebooks) and
**MLflow** (a tracking store + a UI you can actually see). Keep them separate.

## Why not "just Colab"?

Google Colab runs Jupyter, **not marimo** — our notebooks are marimo apps. So
"Colab" really means either converting the notebooks to Jupyter (losing marimo's
reactive model, which is core to the teaching) or using **molab**, marimo's own
free hosted service. Prefer molab.

## Recommended: molab (compute) + DagsHub (hosted MLflow)

This keeps the marimo notebooks and the existing `mlflow.*` code **unchanged** —
you only point the tracking URI at a hosted server.

1. **Compute — molab.** Open a marimo notebook straight from GitHub:
   `https://molab.marimo.io/github/anaynayak/mlops-workshop/blob/main/notebooks/02_train.py`
   (free cloud compute, persistent storage, most packages preinstalled).

2. **Tracking + UI — DagsHub.** Every DagsHub repo auto-provisions a hosted MLflow
   server, so there's nothing to tunnel and the UI is a normal web page that
   survives session resets. In the notebook, before any `mlflow` call:
   ```python
   import os
   os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/<user>/<repo>.mlflow"
   os.environ["MLFLOW_TRACKING_USERNAME"] = "<user>"
   os.environ["MLFLOW_TRACKING_PASSWORD"] = "<token>"   # dagshub.com/user/settings/tokens
   ```
   Our code reads `MLFLOW_TRACKING_URI` (see `scripts/train.py` and
   `src/mlops_workshop/registry.py`), so **no other change is needed** — tracking,
   `log_model`, and the registry API all target DagsHub. View runs at the repo's
   MLflow tab instead of `make mlflow`.

   > Everyone writing to one shared server shares the `nyc-taxi-duration`
   > experiment. For a class, either give each attendee their own DagsHub repo, or
   > vary the experiment name per person (`mlflow.set_experiment(f"nyc-taxi-{name}")`).

**Databricks Free Edition** is an equivalent first-party option (managed MLflow +
registry): set `DATABRICKS_HOST`/`DATABRICKS_TOKEN` and
`mlflow.set_tracking_uri("databricks")`. Slightly more auth setup (a PAT and a
`/Users/<you>/...` experiment path), so DagsHub is the fewer-steps default.

## Not recommended: Colab + tunneled local MLflow

You *can* run a local `mlflow ui` on Colab and expose it with `proxyPort`/ngrok/
cloudflared, plus copy the SQLite db to Google Drive to survive resets. It works,
but it's the most fragile path (proxyPort has no WebSocket support, ngrok needs a
per-attendee token, SQLite-over-Drive can hit `database is locked`). Avoid it for a
non-expert audience — use the hosted-tracker route above instead.
