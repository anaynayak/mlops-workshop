git clone https://github.com/anaynayak/mlops-workshop
cd mlops-workshop
uv run poe data    # installs deps on first run, then downloads the sample
uv run poe sample
uv run poe lab     # launch the marimo notebooks
