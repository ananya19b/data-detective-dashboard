# Data Detective — Statistical Dashboard (No AI)

A transparent, Python/Streamlit dashboard for exploring CSV and Excel datasets. No LLM, no API key, and no machine-learning models.

## Features
- Dataset and column profiling, min/max/mean/std/percentiles
- Missing value counts, charts, and flagged incomplete rows
- Exact duplicate records
- Adjustable IQR univariate outlier detection
- Rare categories (up to 1% frequency, min 2 records)
- Correlation heatmaps, distribution plots, scatterplots and time-series visualization
- Rule-trigger counts for prioritization, reason text and CSV export

Each triggered rule adds one to `anomaly_score`. This is a **count of checks triggered**, not a probability or calibrated anomaly severity. Statistical flags require human verification.

## Run locally
Requires Python 3.10+. Open the folder in VS Code and run:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows instead: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Upload one of the CSV files in `sample_data/` to test.

## Deploy
Push the contents of this folder into a GitHub repository. At https://share.streamlit.io/ deploy the repository with entrypoint `app.py`, then upload a sample file. Do not upload confidential company data to a public demo.

## Notes
IQR is not seasonally adjusted and rare categories can be legitimate. The data-health score is a heuristic, not a regulatory-grade metric. No uploaded data is sent to an LLM.
