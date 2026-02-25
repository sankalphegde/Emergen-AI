# Emergen-AI
Emergency department triage recommendation system built for IE7275 (Data Mining in Engineering, Spring 2026).

## Problem
Prioritize incoming ED visits by criticality so high-risk patients are surfaced earlier for treatment decisions.

## Dataset
This project uses MIMIC-IV ED tables (credentialed access required via PhysioNet):
- `triage.csv`
- `vitalsign.csv`
- `edstays.csv`
- `diagnosis.csv`
- `medrecon.csv`
- `pyxis.csv`

Place raw files under `data/raw/` before running the pipeline.

## Models Compared
- Rule-based baseline triage recommender
- Context-aware logistic regression recommender
- Graph-based recommender (patient-attribute embeddings)
- Pairwise learning-to-rank recommender

## Repository Layout
- `src/` - preprocessing, EDA, and modeling pipelines
- `data/` - raw/processed data (large files are excluded from GitHub)
- `reports/` - metrics, figures, report drafts, and final PDFs
- `notebooks/` - submission notebook

## Quickstart
Run from repo root:

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/eda_visualization.py
python src/triage_pipeline.py
```

## Optional Runtime Controls
Useful for quick iteration on smaller subsets:

```bash
FAST_MODE=1 CONTEXT_MAX_ROWS=120000 PIPELINE_MAX_ROWS=400000 python src/triage_pipeline.py
```

## Main Outputs
- `reports/triage_metrics.json`
- `reports/model_selection_context.png`
- `reports/model_selection_graph.png`
- `reports/eda/eda_report.pdf`
- `reports/ER_Triage_Report_Final.pdf`
- `reports/ER_Triage_Report_VisualDraft.pdf`
- `notebooks/triage_recommender.ipynb`

## Notes
- GitHub rejects files larger than 100MB. Processed data files are kept locally and not pushed.
- Final report metrics should match your Colab run outputs when Colab is the source of truth.
