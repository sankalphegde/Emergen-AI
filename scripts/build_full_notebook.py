#!/usr/bin/env python3
"""Build ER_Triage_Submission_StepwiseOutputs.ipynb with full inline source + stepwise outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
NOTEBOOKS = ROOT / "notebooks"
OUT_PATH = NOTEBOOKS / "ER_Triage_Submission_StepwiseOutputs.ipynb"


def read_src(name):
    path = SRC / name
    return path.read_text()


def md_cell(lines):
    if isinstance(lines, str):
        lines = [lines]
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code_cell(source):
    if isinstance(source, str):
        source = source.strip()
        lines = [line + "\n" for line in source.split("\n")]
    else:
        lines = list(source)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def main():
    data_preprocessing = read_src("data_preprocessing.py")
    feature_engineering = read_src("feature_engineering.py")
    eda_visualization = read_src("eda_visualization.py")
    triage_pipeline = read_src("triage_pipeline.py")

    cells = [
        md_cell([
            "# IE7275 Group Project 1 – Full Code + Stepwise Outputs",
            "",
            "**Project:** Emergency Room Triage Recommendation System (MIMIC-IV ED)",
            "",
            "This notebook includes **full source code** for the project and shows **outputs immediately after each step**:",
            "- Raw data preprocessing → master dataset preview",
            "- Feature engineering → model-ready preview",
            "- EDA → plots inline",
            "- Modeling pipeline → metrics table and model-selection figures",
            "",
            "Run all cells from top to bottom. Outputs appear right after each section.",
        ]),
        md_cell(["## 1. Environment and project root"]),
        code_cell("""from pathlib import Path
import os

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    os.chdir(ROOT.parent)
    ROOT = Path.cwd().resolve()

print("CWD:", ROOT)
print("data/raw exists:", (ROOT / "data" / "raw").exists())
print("src exists:", (ROOT / "src").exists())"""),
        md_cell(["## 2. Data preprocessing (full source)", "File: `src/data_preprocessing.py`"]),
        code_cell(data_preprocessing),
        md_cell(["### Run preprocessing and show output"]),
        code_cell("""# Run the preprocessing logic (definitions are in the cell above)
os.chdir(ROOT)
all_data = load_all_files()
if all_data:
    final_df = preprocess_master(all_data)
    output_file = ROOT / "data" / "processed" / "master_dataset.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print("\\n--- PREPROCESSING COMPLETE ---")
    print("Master file:", output_file)
    print("Shape:", final_df.shape)
    print("\\nFirst 5 rows:")
    final_df.head()"""),
        md_cell(["## 3. Feature engineering (full source)", "File: `src/feature_engineering.py`"]),
        code_cell(feature_engineering),
        md_cell(["### Run feature engineering and show output"]),
        code_cell("""# Run feature engineering (definitions above). Use master from project path.
os.chdir(ROOT)
import pandas as pd
import numpy as np
master_path = ROOT / "data" / "processed" / "master_dataset.csv"
df = pd.read_csv(master_path)
df = calculate_patient_features(df)
df = calculate_context_features(df)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
if len(datetime_cols) > 0:
    df = df.drop(columns=list(datetime_cols))
out_path = ROOT / "data" / "processed" / "model_ready.csv"
df.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S", chunksize=200_000)
print("Feature engineering complete. Model-ready file:", out_path)
print("Shape:", df.shape)
df.head()"""),
        md_cell(["## 4. EDA visualization (full source)", "File: `src/eda_visualization.py`"]),
        code_cell(eda_visualization),
        md_cell(["### Run EDA and show key plots inline"]),
        code_cell("""# Run EDA (uses DATA_PATH; ensure we're in project root). Then display figures.
import os
os.chdir(ROOT)
main()

from IPython.display import Image, display
eda_dir = ROOT / "reports" / "eda"
for name in ["count_acuity.png", "count_disposition.png", "count_arrival_hour.png", "corr_heatmap.png", "boxplot_vitals.png"]:
    p = eda_dir / name
    if p.exists():
        display(Image(filename=str(p)))"""),
        md_cell(["## 5. Modeling and evaluation pipeline (full source)", "File: `src/triage_pipeline.py`"]),
        code_cell(triage_pipeline),
        md_cell(["### Run pipeline and show metrics + figures"]),
        code_cell("""# Run full triage pipeline (definitions in cell above)
os.chdir(ROOT)  # ensure paths in triage_pipeline.main() resolve correctly
main()

# Load and display metrics
import json
from IPython.display import Image, display
metrics_path = ROOT / "reports" / "triage_metrics.json"
with open(metrics_path) as f:
    metrics = json.load(f)

rows = []
for model in ["baseline", "context", "graph", "pairwise"]:
    d = metrics.get(model, {})
    rows.append({
        "Model": model,
        "Accuracy": d.get("accuracy"),
        "ROC-AUC": d.get("roc_auc"),
        "PR-AUC": d.get("pr_auc"),
        "F1": d.get("f1"),
        "F2": d.get("f2"),
        "Precision@5": d.get("k=5", {}).get("precision@k"),
        "Recall@5": d.get("k=5", {}).get("recall@k"),
        "Precision@10": d.get("k=10", {}).get("precision@k"),
        "Recall@10": d.get("k=10", {}).get("recall@k"),
    })
pd.DataFrame(rows)

# Model selection figures
for name in ["model_selection_context.png", "model_selection_graph.png"]:
    p = ROOT / "reports" / name
    if p.exists():
        display(Image(filename=str(p)))"""),
        md_cell([
            "### Key findings",
            "- **Context-aware model** is strongest on ROC-AUC, PR-AUC, F1/F2, and accuracy.",
            "- **Baseline** has decent top-k relevance but weaker global metrics.",
            "- **Graph** and **pairwise** are near chance in this setup.",
        ]),
        md_cell(["## 6. Artifact check"]),
        code_cell("""artifacts = [
    ROOT / "data" / "processed" / "master_dataset.csv",
    ROOT / "data" / "processed" / "model_ready.csv",
    ROOT / "reports" / "triage_metrics.json",
    ROOT / "reports" / "model_selection_context.png",
    ROOT / "reports" / "model_selection_graph.png",
    ROOT / "reports" / "eda" / "eda_report.pdf",
]
for a in artifacts:
    print(a, "->", "OK" if a.exists() else "MISSING")"""),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "cells": cells,
    }

    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print("Wrote", OUT_PATH)


if __name__ == "__main__":
    main()
