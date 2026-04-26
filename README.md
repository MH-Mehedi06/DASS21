# DASS-21 (Depression, Anxiety, Stress Scale) — ML + Web App

This folder contains a small machine learning project around the **DASS-21** questionnaire:

- A notebook for exploration and modeling: `DASS21.ipynb`
- A simple Flask web app to take the questionnaire and see results: `app.py`
- Reusable Python modules for data cleaning, scoring, clustering, and plotting: `src/`

> **Important**: This project is for educational/analytics purposes and is **not** a clinical diagnostic tool.

## What it does

- **Loads and cleans** a DASS-21 dataset (Excel or CSV)
- **Computes DASS-21 scores** (Depression / Anxiety / Stress) and severity labels
- **Clusters respondents** using K-Means (and optional validation utilities using GMM + t-SNE)
- **Provides a web UI** where a user answers the 21 items (0–3) and gets:
  - Severity labels and scores (only non-`Normal` are shown; otherwise “All Levels Normal”)
  - A K-Means cluster “Group” prediction (statistical grouping)

## Project structure

- **`DASS21.ipynb`**: end-to-end analysis (cleaning → scoring → clustering/plots)
- **`app.py`**: Flask app (questionnaire form + result page)
- **`src/data.py`**: `load_and_clean_data()` (renames columns, drops duplicates, removes straight-liners)
- **`src/features.py`**: `calculate_scores()` + severity label helpers
- **`src/clustering.py`**: scaling + `run_kmeans()`, `run_gmm()`, t-SNE visualization helper
- **`src/visualization.py`**: correlation + severity distribution plots
- **`templates/`**: `index.html`, `result.html`
- **`static/style.css`**: styling for the web UI

## Dataset file (required)

Both the notebook and the Flask app expect a dataset file at:

- **`DASS 21 Dataset.xlsx`** (default), or
- a CSV file (if you update the path in code)

Place the dataset file **in this folder** (same directory as `app.py`).

### Expected columns

`src/data.py` assumes the dataset contains:

- The first two columns: `Gender`, `Age` (or equivalent positions)
- Then 21 question columns mapped to `Q1` … `Q21`

If your file has a different layout, adjust the renaming logic in `src/data.py`.

## Setup

Use Python 3.9+ (3.10+ recommended).

Install the main dependencies:

```bash
pip install flask pandas scikit-learn matplotlib seaborn openpyxl
```

Notes:
- `openpyxl` is needed to read `.xlsx`.
- If you only use CSV, you can skip `openpyxl`.

## Run the notebook

Open `DASS21.ipynb` in Jupyter / VS Code and run cells.

Make sure `DASS 21 Dataset.xlsx` is in the same folder so the notebook can load it (or update the path inside the notebook).

## Run the web app

From this folder:

```bash
python app.py
```

Then open the local URL printed in the terminal (Flask default is `http://127.0.0.1:5000`).

### How the web app scores and clusters

- Your responses are scored using standard DASS-21 item keys in `src/features.py`.
- A K-Means model is trained **on startup** from the dataset file (`DASS 21 Dataset.xlsx`).
- Your 21 responses are scaled with a `StandardScaler` fit on the dataset, then assigned a cluster group.

If the dataset file is missing, clustering will show as `Unknown`.

## Optional: validate clustering quality

This script compares K-Means vs GMM and writes plots:

```bash
python validate_clusters.py
```

It will save images to a `plots/` folder (created automatically).

## Troubleshooting

- **Excel file won’t load**: install `openpyxl` (`pip install openpyxl`) and ensure the dataset filename matches exactly.
- **`DASS 21 Dataset.xlsx` not found**: put the file next to `app.py` or change `DATA_PATH` in `app.py` and `validate_clusters.py`.
- **Import errors for `src.*`**: run commands from this folder (so `src/` is importable).

## License

See `LICENSE`.