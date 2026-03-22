# MLOps Workshop

2-hour introductory MLOps workshop for platform engineers, data engineers, and data scientists new to MLOps.

## Tech Stack

- Python 3.12
- uv for dependency management
- marimo for notebooks
- pandas, scikit-learn, MLflow
- Makefile for commands

## Dataset

NYC Taxi Trips - regression task (predict trip duration)

## Commands

- `make setup` - install dependencies
- `make data` - download dataset from WORKSHOP_SAMPLE_URL
- `make lab` - launch marimo notebooks

## Guidelines

- Keep commits small and atomic
- Use plain commit messages without prefixes (no feat:/fix:/refactor:)
- Optimize for workshop outcome, not architecture purity
- Keep setup low-friction for attendees
