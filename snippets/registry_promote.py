from mlops_workshop import registry

# 1. Find a candidate — runs sorted best-first
registry.find_runs(metric="rmse")            # eyeball it, copy a run_id

# 2. Register that run's model as a new version (this is the API — no clicking in the UI)
mv = registry.register_run("<run_id>")       # -> version N

# 3. Stage it as the challenger
registry.set_alias("challenger", mv.version)

# 4. It beats the champion? Promote it.
registry.set_alias("champion", mv.version)

# The consumer — 04_inference.py — stops caring about files:
model = registry.load_model("champion")      # always the live model
