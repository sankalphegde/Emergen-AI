# Emergen-AI
Predicting patient urgency through emergent data patterns

## Project Overview
This project builds an emergency department triage recommendation system using the MIMIC-IV ED dataset.
We frame triage as a ranking problem and compare:
- Rule-based baseline
- Context-aware model
- Graph-based model
- Pairwise learning-to-rank model

## Repo Structure
- `src/`: pipeline scripts
- `data/`: raw and processed datasets (not tracked)
- `reports/`: EDA + metrics outputs (not tracked)

## How to Run
1. Preprocess & merge:
```bash
python src/data_preprocessing.py
```

2. Feature engineering:
```bash
python src/feature_engineering.py
```

3. EDA visualizations:
```bash
python src/eda_visualization.py
```

4. Model training + evaluation:
```bash
python src/triage_pipeline.py
```

## Key Outputs
- `reports/eda/eda_report.pdf`
- `reports/triage_metrics.json`
- `reports/model_selection_context.png`
- `reports/model_selection_graph.png`
