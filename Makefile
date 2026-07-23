.PHONY: setup data sample lab train infer feature-store serve test mlflow slides

# Task definitions live in pyproject.toml under [tool.poe.tasks].
# These targets are thin wrappers so `make <target>` still works on
# machines that have make; the cross-platform entry point (Windows too)
# is `uv run poe <target>`.

setup:
	uv sync

data:
	uv run poe data

sample:
	uv run poe sample

lab:
	uv run poe lab

train:
	uv run poe train

infer:
	uv run poe infer

feature-store:
	uv run python scripts/build_feature_store.py

serve:
	uv run python scripts/serve.py

test:
	uv run poe test

mlflow:
	uv run poe mlflow

slides:
	uv run poe slides
