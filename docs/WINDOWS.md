# Windows setup and gotchas

The workshop runs fine on Windows, but a few setup mechanics differ from
macOS/Linux. None are blockers — this page lists each gotcha and its fix. If you
hit something not covered here, add it.

## The one thing to know: there is no `make` on Windows

Windows doesn't ship GNU `make`. **Don't install it — you don't need it.** Every
`make <task>` in the README is a thin wrapper around a cross-platform task runner.
Run the task runner directly:

| Instead of | Run |
|---|---|
| `make setup` | `uv sync` |
| `make data` | `uv run poe data` |
| `make sample` | `uv run poe sample` |
| `make lab` | `uv run poe lab` |
| `make train` | `uv run poe train` |
| `make infer` | `uv run poe infer` |
| `make mlflow` | `uv run poe mlflow` |
| `make test` | `uv run poe test` |

(The full task list lives in `pyproject.toml` under `[tool.poe.tasks]`. Note
`uv run poe setup-feature-store` is poe-only — there's no `make` wrapper for it.)

## Quick start (PowerShell)

```powershell
# 1. Install uv (Astral's Python package manager)
winget install --id=astral-sh.uv  -e
#   ...or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Open a NEW terminal so uv is on PATH, then:
git clone https://github.com/anaynayak/mlops-workshop
cd mlops-workshop
uv sync
uv run poe data
uv run poe sample
uv run poe lab
```

## Gotchas and fixes

1. **uv "not found" right after install.** The installer adds uv to PATH but your
   current terminal won't see it. Open a new terminal (or log out/in). Verify with
   `uv --version`.

2. **Enable long paths.** MLflow writes deeply nested artifact directories under
   `mlruns/`; Windows' legacy 260-character path limit can make those writes fail.
   Enable long paths once (admin PowerShell):
   ```powershell
   Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
     -Name LongPathsEnabled -Value 1
   ```
   and set `git config --global core.longpaths true` before cloning.

3. **Firewall prompt on `mlflow ui` / `marimo`.** Both start a local web server
   bound to `localhost`. Windows Defender will pop a "allow access?" dialog the
   first time — allow it (private networks is enough). Then open the printed URL
   (MLflow: http://localhost:5000, marimo: the URL it prints).

4. **Setting env vars differs from bash.** Anywhere the docs show
   `WORKSHOP_SAMPLE_URL=... make data` (bash syntax), use PowerShell syntax:
   ```powershell
   $env:WORKSHOP_SAMPLE_URL = "https://.../my_sample.parquet"
   uv run poe data
   ```
   Same for `MLFLOW_TRACKING_URI` if you point at a hosted server.

5. **The heavy libraries are prebuilt — no compiler needed.** `xgboost`,
   `scikit-learn`, `pyarrow`, `scipy`, and `cryptography` all install as prebuilt
   Windows wheels via `uv sync`. If `uv sync` seems to hang, it's downloading them
   the first time — give it a minute.

6. **Feature store (Feast) is optional.** The `feature-store` dependency group is
   the most likely source of Windows friction and is **not** installed by default.
   Only run `uv run poe setup-feature-store` if you're doing that section.

## Notebook mechanics (any OS, but bites people on Windows too)

- **marimo has no "redeclare".** A variable is defined in exactly one cell. When
  you add the MLflow tracking code (`snippets/tracking.py`) to `02_train.py`, that
  one cell defines `model` and `metrics` — so it **replaces** the separate train
  and evaluate cells. Don't leave the old definitions around, or marimo flags a
  redeclaration error.

## Escape hatch: run it in the cloud

If a machine genuinely won't cooperate and you don't want to burn workshop time on
it, you can run the notebooks in the browser with a hosted MLflow and zero local
install. See [`CLOUD.md`](./CLOUD.md).
