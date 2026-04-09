.PHONY: setup data sample lab train infer test mlflow slides

setup:
	uv sync

data:
	@if [ -f data/raw/nyc_taxi.parquet ]; then \
		echo "Data already exists"; \
	elif [ -z "$$WORKSHOP_SAMPLE_URL" ]; then \
		echo "Error: WORKSHOP_SAMPLE_URL must be set"; \
		exit 1; \
	else \
		mkdir -p data/raw; \
		wget -O data/raw/nyc_taxi.parquet "$$WORKSHOP_SAMPLE_URL"; \
	fi

sample:
	uv run python scripts/sample_data.py

lab:
	uv run marimo edit notebooks/

train:
	uv run python scripts/train.py

infer:
	uv run python scripts/infer.py

test:
	uv run pytest -v

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

slides:
	cd slides && npm run dev