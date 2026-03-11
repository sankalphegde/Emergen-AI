# Emergen-AI
Emergency department triage recommendation system for IE7275 (Data Mining in Engineering, Spring 2026).

## Project Summary
This project ranks incoming ED visits by clinical risk to support faster prioritization of high‑acuity patients. It integrates multiple MIMIC‑IV ED tables into a visit‑level dataset, performs EDA, trains a baseline and advanced recommenders, and evaluates relevance and discrimination metrics.

## Dataset
MIMIC‑IV ED (credentialed access required via PhysioNet). Raw inputs expected under `data/raw/`:
- `triage.csv`
- `vitalsign.csv`
- `edstays.csv`
- `diagnosis.csv`
- `medrecon.csv`
- `pyxis.csv`

## Models Implemented
- Rule‑based baseline triage recommender
- Context‑aware logistic regression
- Graph‑based recommender (patient‑attribute embeddings)
- Pairwise learning‑to‑rank model

## Evaluation Metrics
Precision@k, Recall@k, ROC‑AUC, PR‑AUC, F1, F2, Accuracy. Thresholds are selected on validation data (F2‑oriented).

## Deliverables
- PDF report: `reports/ER_Triage_Report_VisualDraft.pdf`
- Submission notebook: `notebooks/ER_Triage_Submission_Final.ipynb`

## Repository Layout
- `src/` - preprocessing, feature engineering, EDA, modeling
- `data/` - raw/processed data (large files excluded from GitHub)
- `reports/` - figures, metrics, report PDF
- `notebooks/` - submission notebooks

## Run Pipeline (Local)
From repo root:

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/eda_visualization.py
python src/triage_pipeline.py
```

## Notes
- Processed datasets are not committed due to GitHub size limits and data access restrictions.
- The submission notebook is meant to be run locally where the raw data is available.
