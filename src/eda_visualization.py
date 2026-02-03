import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


DATA_PATH = "data/processed/master_dataset.csv"
OUT_DIR = "reports/eda"


def unify_vitals(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("temperature", "temperature_y", "temperature_x"),
        ("heartrate", "heartrate_y", "heartrate_x"),
        ("resprate", "resprate_y", "resprate_x"),
        ("o2sat", "o2sat_y", "o2sat_x"),
        ("sbp", "sbp_y", "sbp_x"),
        ("dbp", "dbp_y", "dbp_x"),
        ("pain", "pain_y", "pain_x"),
    ]
    for base, y_col, x_col in pairs:
        if y_col in df.columns and x_col in df.columns:
            df[base] = df[y_col].combine_first(df[x_col])
        elif y_col in df.columns:
            df[base] = df[y_col]
        elif x_col in df.columns:
            df[base] = df[x_col]
    return df


def add_time_features(df: pd.DataFrame, time_col: str = "intime") -> pd.DataFrame:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["arrival_hour"] = df[time_col].dt.hour
    df["arrival_day"] = df[time_col].dt.dayofweek
    return df


def save_plot(fig, filename: str, pdf: PdfPages = None):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, col: str, title: str, filename: str, pdf: PdfPages = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)
    save_plot(fig, filename, pdf)


def plot_count(df: pd.DataFrame, col: str, title: str, filename: str, top_n: int = 10, pdf: PdfPages = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[col].fillna("UNKNOWN").value_counts().head(top_n)
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    save_plot(fig, filename, pdf)


def plot_missingness(df: pd.DataFrame, filename: str, pdf: PdfPages = None):
    miss = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=miss.values[:20], y=miss.index[:20], ax=ax)
    ax.set_title("Top 20 Missingness Rates")
    ax.set_xlabel("Missing Fraction")
    save_plot(fig, filename, pdf)


def plot_correlation(df: pd.DataFrame, cols: List[str], filename: str, pdf: PdfPages = None):
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap (Vitals + Features)")
    save_plot(fig, filename, pdf)


def plot_boxplots(df: pd.DataFrame, cols: List[str], filename: str, pdf: PdfPages = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df[cols], orient="h", ax=ax)
    ax.set_title("Vital Sign Boxplots (Outliers)")
    save_plot(fig, filename, pdf)


def plot_violin_by_acuity(df: pd.DataFrame, col: str, filename: str, pdf: PdfPages = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(x="acuity", y=col, data=df, ax=ax)
    ax.set_title(f"{col} by Acuity")
    save_plot(fig, filename, pdf)


def plot_pairplot(df: pd.DataFrame, cols: List[str], filename: str):
    pp = sns.pairplot(df[cols].dropna())
    pp.fig.suptitle("Pairplot of Key Vitals", y=1.02)
    path = os.path.join(OUT_DIR, filename)
    pp.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(pp.fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = unify_vitals(df)
    df = add_time_features(df, "intime")

    pdf_path = os.path.join(OUT_DIR, "eda_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Basic distributions
        plot_distribution(df, "temperature", "Temperature Distribution", "dist_temperature.png", pdf)
        plot_distribution(df, "heartrate", "Heart Rate Distribution", "dist_heartrate.png", pdf)
        plot_distribution(df, "resprate", "Respiratory Rate Distribution", "dist_resprate.png", pdf)
        plot_distribution(df, "o2sat", "O2 Saturation Distribution", "dist_o2sat.png", pdf)
        plot_distribution(df, "sbp", "Systolic BP Distribution", "dist_sbp.png", pdf)
        plot_distribution(df, "dbp", "Diastolic BP Distribution", "dist_dbp.png", pdf)

        # Categorical counts
        plot_count(df, "disposition", "Disposition Counts", "count_disposition.png", pdf=pdf)
        plot_count(df, "acuity", "Acuity Counts", "count_acuity.png", pdf=pdf)
        plot_count(df, "arrival_transport", "Arrival Transport", "count_transport.png", pdf=pdf)
        plot_count(df, "chiefcomplaint", "Top Chief Complaints", "count_chiefcomplaint.png", top_n=15, pdf=pdf)

        # Time patterns
        plot_count(df, "arrival_hour", "Arrivals by Hour", "count_arrival_hour.png", top_n=24, pdf=pdf)
        plot_count(df, "arrival_day", "Arrivals by Day of Week (0=Mon)", "count_arrival_day.png", top_n=7, pdf=pdf)

        # Missingness + correlation
        plot_missingness(df, "missingness_top20.png", pdf)
        corr_cols = [
            "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
            "num_prior_cond", "num_home_meds", "er_meds_given"
        ]
        corr_cols = [c for c in corr_cols if c in df.columns]
        plot_correlation(df, corr_cols, "corr_heatmap.png", pdf)

        # Outlier boxplots
        vitals_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
        vitals_cols = [c for c in vitals_cols if c in df.columns]
        plot_boxplots(df, vitals_cols, "boxplot_vitals.png", pdf)

        # Violin plots by acuity
        for col in ["heartrate", "resprate", "o2sat", "sbp"]:
            if col in df.columns:
                plot_violin_by_acuity(df, col, f"violin_{col}_by_acuity.png", pdf)

    # Pairplot (saved as PNG only to keep PDF size reasonable)
    pair_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
    pair_cols = [c for c in pair_cols if c in df.columns]
    if len(pair_cols) >= 2:
        plot_pairplot(df, pair_cols, "pairplot_vitals.png")

    # Summary table
    summary = df.describe(include="all").transpose()
    summary.to_csv(os.path.join(OUT_DIR, "summary_stats.csv"))

    print(f"EDA plots saved to {OUT_DIR}")
    print(f"PDF report saved to {pdf_path}")


if __name__ == "__main__":
    main()
