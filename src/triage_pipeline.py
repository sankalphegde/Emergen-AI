import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt


@dataclass
class Config:
    data_path: str = "data/processed/master_dataset.csv"
    output_dir: str = "reports"
    time_col: str = "intime"
    test_size: float = 0.2
    random_state: int = 42
    k_values: Tuple[int, ...] = (5, 10)
    decision_threshold: float = 0.4
    label_strategy: str = "acuity_le2"
    threshold_metric: str = "f2"
    threshold_min: float = 0.1
    threshold_max: float = 0.9
    threshold_step: float = 0.05


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def unify_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer *_y columns (from vitals) and fallback to *_x (from triage).
    Produces unified columns without suffixes.
    """
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


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["arrival_hour"] = df[time_col].dt.hour
    df["arrival_day"] = df[time_col].dt.dayofweek
    return df


def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    df["shock_index"] = df["heartrate"] / df["sbp"]
    df["pulse_pressure"] = df["sbp"] - df["dbp"]
    df["complexity_score"] = df["num_prior_cond"].fillna(0) + df["num_home_meds"].fillna(0)
    return df


def simulate_context(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    More realistic congestion simulation:
    - arrivals_per_hour: count per hour
    - rolling_arrivals_6h: rolling 6-hour arrivals
    - active_patients: number of ongoing ED stays at arrival time
    - bed_capacity: 85th percentile of active patients
    - congestion_score: active_patients / bed_capacity (capped)
    - resource_availability: inverse of congestion
    - wait_time_proxy: backlog * 5 minutes
    """
    df = df.sort_values(time_col).copy()
    df["arrival_hour_bucket"] = df[time_col].dt.floor("h")
    arrivals = df.groupby("arrival_hour_bucket").size().rename("arrivals_per_hour")
    df = df.merge(arrivals, on="arrival_hour_bucket", how="left")

    rolling = arrivals.rolling(window=6, min_periods=1).sum().rename("rolling_arrivals_6h")
    df = df.merge(rolling, on="arrival_hour_bucket", how="left")

    # Active patient count at arrival
    times = df[time_col].values
    outtimes = pd.to_datetime(df["outtime"], errors="coerce").values
    events = []
    for t_in, t_out in zip(times, outtimes):
        if pd.isna(t_in) or pd.isna(t_out):
            continue
        events.append((t_in, 1))
        events.append((t_out, -1))
    events.sort(key=lambda x: x[0])

    active_counts = np.zeros(len(df), dtype=int)
    order = np.argsort(times)
    running = 0
    ei = 0
    for idx in order:
        t = times[idx]
        while ei < len(events) and events[ei][0] < t:
            running += events[ei][1]
            ei += 1
        active_counts[idx] = max(running, 0)

    df["active_patients"] = active_counts
    bed_capacity = np.percentile(active_counts, 85) if len(active_counts) else 1.0
    bed_capacity = max(bed_capacity, 1.0)
    df["bed_capacity"] = bed_capacity
    df["congestion_score"] = np.minimum(df["active_patients"] / bed_capacity, 1.5)
    df["resource_availability"] = np.maximum(0.0, 1.0 - df["congestion_score"])
    df["wait_time_proxy"] = np.maximum(0, df["active_patients"] - bed_capacity) * 5
    return df


def create_label(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Label strategies:
    - acuity_le2: critical if acuity <= 2
    - admit_or_transfer: critical if disposition in {ADMITTED, TRANSFER}
    - combined: acuity<=2 OR admit/transfer
    """
    df = df.copy()
    if strategy == "acuity_le2" and "acuity" in df.columns:
        df["is_critical"] = (df["acuity"].fillna(99) <= 2).astype(int)
    elif strategy == "combined" and "acuity" in df.columns:
        critical_dispo = {"ADMITTED", "TRANSFER"}
        df["is_critical"] = (
            (df["acuity"].fillna(99) <= 2) | (df["disposition"].isin(critical_dispo))
        ).astype(int)
    else:
        critical_dispo = {"ADMITTED", "TRANSFER"}
        df["is_critical"] = df["disposition"].isin(critical_dispo).astype(int)
    return df


def time_split_three(
    df: pd.DataFrame, time_col: str, train_frac: float = 0.6, val_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_col).copy()
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def rule_based_score(row: pd.Series) -> int:
    """
    Rule-based triage score using vital sign thresholds.
    Higher score = higher urgency.
    """
    score = 0
    if row["sbp"] < 90:
        score += 2
    if row["o2sat"] < 92:
        score += 2
    if row["heartrate"] > 130:
        score += 1
    if row["resprate"] > 30:
        score += 1
    if row["temperature"] > 103 or row["temperature"] < 95:
        score += 1
    return score


def evaluate_at_k(df: pd.DataFrame, score_col: str, label_col: str, group_col: str, k: int) -> Dict[str, float]:
    """
    Compute Recall@k and Precision@k by grouping (e.g., hour buckets).
    """
    recalls = []
    precisions = []
    for _, group in df.groupby(group_col):
        if group[label_col].sum() == 0:
            continue
        ranked = group.sort_values(score_col, ascending=False)
        top_k = ranked.head(k)
        recall = top_k[label_col].sum() / group[label_col].sum()
        precision = top_k[label_col].sum() / min(k, len(group))
        recalls.append(recall)
        precisions.append(precision)
    if not recalls:
        return {"recall@k": 0.0, "precision@k": 0.0}
    return {
        "recall@k": float(np.mean(recalls)),
        "precision@k": float(np.mean(precisions)),
    }


def fbeta_score_safe(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta2 = beta ** 2
    if precision == 0 and recall == 0:
        return 0.0
    return (1 + beta2) * (precision * recall) / (beta2 * precision + recall)


def find_best_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    t_min: float,
    t_max: float,
    t_step: float,
) -> Tuple[float, float]:
    best_t = t_min
    best_score = -1.0
    thresholds = np.arange(t_min, t_max + 1e-9, t_step)
    for t in thresholds:
        y_pred = (scores > t).astype(int)
        if metric == "f2":
            score = fbeta_score_safe(y_true, y_pred, beta=2.0)
        elif metric == "f1":
            score = f1_score(y_true, y_pred)
        else:
            score = fbeta_score_safe(y_true, y_pred, beta=2.0)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


def build_pairwise_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    max_pairs_per_group: int = 200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X_list = []
    y_list = []
    for _, group in df.groupby(group_col):
        pos = group[group["is_critical"] == 1]
        neg = group[group["is_critical"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pairs = []
        for _, p in pos.iterrows():
            for _, n in neg.iterrows():
                pairs.append((p, n))
        if len(pairs) > max_pairs_per_group:
            idx = rng.choice(len(pairs), size=max_pairs_per_group, replace=False)
            pairs = [pairs[i] for i in idx]
        for p, n in pairs:
            X_list.append(p[feature_cols].values - n[feature_cols].values)
            y_list.append(1)
    if not X_list:
        return np.empty((0, len(feature_cols))), np.empty((0,))
    return np.vstack(X_list), np.array(y_list)


def train_pairwise_ranker(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
) -> Tuple[StandardScaler, LogisticRegression | None]:
    X_train, y_train = build_pairwise_dataset(train, feature_cols, group_col)
    scaler = StandardScaler()
    if X_train.shape[0] == 0:
        scaler.fit(train[feature_cols].values)
        return scaler, None
    X_train = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", fit_intercept=False)
    model.fit(X_train, y_train)
    return scaler, model


def score_pairwise(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    model: LogisticRegression | None,
) -> np.ndarray:
    X = df[feature_cols].values
    X = scaler.transform(X)
    if model is None:
        return np.zeros(len(df))
    return model.decision_function(X)


def train_context_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline, Dict[str, float], Dict[str, np.ndarray]]:
    X_train = train[feature_cols + cat_cols]
    y_train = train["is_critical"]
    X_test = test[feature_cols + cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), feature_cols),
        ]
    )

    base_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = Pipeline(steps=[("pre", preprocessor), ("model", base_model)])

    param_grid = {
        "model__C": [0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    }
    grid = GridSearchCV(
        clf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    test = test.copy()
    test["context_score"] = clf.predict_proba(X_test)[:, 1]
    train = train.copy()
    train["context_score"] = clf.predict_proba(X_train)[:, 1]
    return train, test, clf, grid.best_params_, grid.cv_results_


def build_graph_embeddings(df: pd.DataFrame, dim: int = 16) -> pd.DataFrame:
    """
    Simple graph embedding using SVD on patient-attribute bipartite matrix.
    Nodes: patients; attributes: chief complaint, arrival_transport, race.
    """
    attrs = ["chiefcomplaint", "arrival_transport", "race"]
    df = df.copy()
    df["patient_id"] = df["stay_id"].astype(str)

    for col in attrs:
        df[col] = df[col].fillna("UNKNOWN")

    attr_values = []
    for col in attrs:
        vals = df[col].astype(str).unique().tolist()
        attr_values.extend([f"{col}:{v}" for v in vals])
    attr_index = {v: i for i, v in enumerate(sorted(attr_values))}

    patient_index = {pid: i for i, pid in enumerate(df["patient_id"].unique())}
    mat = np.zeros((len(patient_index), len(attr_index)), dtype=np.float32)

    for _, row in df.iterrows():
        p = patient_index[row["patient_id"]]
        for col in attrs:
            key = f"{col}:{row[col]}"
            mat[p, attr_index[key]] = 1.0

    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    emb = u[:, :dim] * s[:dim]

    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(dim)])
    emb_df["patient_id"] = list(patient_index.keys())
    return emb_df


def train_graph_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    emb_df: pd.DataFrame,
    base_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[StandardScaler, LogisticRegression] | None, Dict[str, float], Dict[str, np.ndarray]]:
    train = train.copy()
    test = test.copy()
    train["patient_id"] = train["stay_id"].astype(str)
    test["patient_id"] = test["stay_id"].astype(str)

    train = train.merge(emb_df, on="patient_id", how="left")
    test = test.merge(emb_df, on="patient_id", how="left")

    emb_cols = [c for c in train.columns if c.startswith("emb_")]
    feature_cols = base_cols + emb_cols

    X_train = train[feature_cols].fillna(0)
    y_train = train["is_critical"]
    X_test = test[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }
    grid = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
    )
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_
    train["graph_score"] = model.predict_proba(X_train_scaled)[:, 1]
    test["graph_score"] = model.predict_proba(X_test_scaled)[:, 1]
    return train, test, (scaler, model), grid.best_params_, grid.cv_results_


def plot_model_selection(cv_results: Dict[str, np.ndarray], title: str, out_path: str):
    params = cv_results.get("params", [])
    mean_scores = cv_results.get("mean_test_score", [])
    if not params or len(mean_scores) == 0:
        return

    c_vals = [p.get("model__C", p.get("C")) for p in params]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c_vals, mean_scores, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Mean CV ROC-AUC")
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = Config()

    # Step 1: Load data
    df = load_data(cfg.data_path)

    # Step 2: Unify vitals + time features
    df = unify_vitals(df)
    df = add_time_features(df, cfg.time_col)

    # Step 3: Feature engineering
    df = add_patient_features(df)
    df = simulate_context(df, cfg.time_col)

    # Step 4: Label creation
    df = create_label(df, cfg.label_strategy)

    # Step 5: Basic cleanup
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Step 6: Train/val/test split
    train, val, test = time_split_three(df, cfg.time_col)

    # Step 7: Baseline rule-based triage
    test = test.copy()
    val = val.copy()
    test["rule_score"] = test.apply(rule_based_score, axis=1)
    val["rule_score"] = val.apply(rule_based_score, axis=1)

    # Step 8: Context-aware model
    numeric_features = [
        "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
        "shock_index", "pulse_pressure", "complexity_score",
        "arrivals_per_hour", "rolling_arrivals_6h", "active_patients",
        "bed_capacity", "congestion_score", "resource_availability", "wait_time_proxy",
        "arrival_hour", "arrival_day"
    ]
    categorical_features = ["gender", "race", "arrival_transport", "chiefcomplaint"]
    train, test, context_model, context_params, context_cv = train_context_model(
        train, test, numeric_features, categorical_features
    )
    # Score validation set for threshold tuning
    X_val = val[numeric_features + categorical_features]
    val["context_score"] = context_model.predict_proba(X_val)[:, 1]

    # Step 9: Graph-based model
    emb_df = build_graph_embeddings(df)
    base_graph_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "congestion_score"]
    train, test, graph_model, graph_params, graph_cv = train_graph_model(
        train, test, emb_df, base_graph_cols
    )
    val = val.copy()
    val["patient_id"] = val["stay_id"].astype(str)
    val = val.merge(emb_df, on="patient_id", how="left")
    emb_cols = [c for c in val.columns if c.startswith("emb_")]
    val_graph_features = base_graph_cols + emb_cols
    X_val_graph = val[val_graph_features].fillna(0)
    if graph_model is not None:
        graph_scaler, graph_clf = graph_model
        X_val_graph_scaled = graph_scaler.transform(X_val_graph)
        val["graph_score"] = graph_clf.predict_proba(X_val_graph_scaled)[:, 1]
    else:
        val["graph_score"] = 0.0

    # Step 9b: Pairwise ranking model (numeric features only)
    pairwise_features = [
        "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
        "shock_index", "pulse_pressure", "complexity_score",
        "congestion_score", "active_patients", "wait_time_proxy",
        "arrival_hour", "arrival_day",
    ]
    scaler, pairwise_model = train_pairwise_ranker(train, val, pairwise_features, "arrival_hour_bucket")
    val["pairwise_score"] = score_pairwise(val, pairwise_features, scaler, pairwise_model)
    test["pairwise_score"] = score_pairwise(test, pairwise_features, scaler, pairwise_model)

    # Step 10: Evaluate ranking metrics
    metrics = {
        "config": {
            "label_strategy": cfg.label_strategy,
            "threshold_metric": cfg.threshold_metric,
            "threshold_range": [cfg.threshold_min, cfg.threshold_max, cfg.threshold_step],
        },
        "baseline": {},
        "context": {"best_params": context_params},
        "graph": {"best_params": graph_params},
        "pairwise": {},
    }
    group_col = "arrival_hour_bucket"
    for k in cfg.k_values:
        metrics["baseline"][f"k={k}"] = evaluate_at_k(test, "rule_score", "is_critical", group_col, k)
        metrics["context"][f"k={k}"] = evaluate_at_k(test, "context_score", "is_critical", group_col, k)
        metrics["graph"][f"k={k}"] = evaluate_at_k(test, "graph_score", "is_critical", group_col, k)
        metrics["pairwise"][f"k={k}"] = evaluate_at_k(test, "pairwise_score", "is_critical", group_col, k)

    # Step 10b: Threshold tuning for critical-prioritized classification
    val_true = val["is_critical"].values
    ctx_t, ctx_val_score = find_best_threshold(
        val["context_score"].values,
        val_true,
        cfg.threshold_metric,
        cfg.threshold_min,
        cfg.threshold_max,
        cfg.threshold_step,
    )
    g_t, g_val_score = find_best_threshold(
        val["graph_score"].values,
        val_true,
        cfg.threshold_metric,
        cfg.threshold_min,
        cfg.threshold_max,
        cfg.threshold_step,
    )
    p_t, p_val_score = find_best_threshold(
        val["pairwise_score"].values,
        val_true,
        cfg.threshold_metric,
        cfg.threshold_min,
        cfg.threshold_max,
        cfg.threshold_step,
    )
    metrics["context"]["best_threshold"] = ctx_t
    metrics["context"]["best_threshold_score"] = ctx_val_score
    metrics["graph"]["best_threshold"] = g_t
    metrics["graph"]["best_threshold_score"] = g_val_score
    metrics["pairwise"]["best_threshold"] = p_t
    metrics["pairwise"]["best_threshold_score"] = p_val_score

    # Step 10c: Classification metrics for model selection
    y_true = test["is_critical"].values
    metrics["baseline"]["roc_auc"] = float(roc_auc_score(y_true, test["rule_score"]))
    metrics["baseline"]["pr_auc"] = float(average_precision_score(y_true, test["rule_score"]))
    metrics["baseline"]["f1"] = float(f1_score(y_true, (test["rule_score"] > 0).astype(int)))
    metrics["baseline"]["accuracy"] = float(
        accuracy_score(y_true, (test["rule_score"] > 0).astype(int))
    )

    metrics["context"]["roc_auc"] = float(roc_auc_score(y_true, test["context_score"]))
    metrics["context"]["pr_auc"] = float(average_precision_score(y_true, test["context_score"]))
    metrics["context"]["f1"] = float(f1_score(y_true, (test["context_score"] > ctx_t).astype(int)))
    metrics["context"]["f2"] = float(
        fbeta_score_safe(y_true, (test["context_score"] > ctx_t).astype(int), beta=2.0)
    )
    metrics["context"]["accuracy"] = float(
        accuracy_score(y_true, (test["context_score"] > ctx_t).astype(int))
    )

    metrics["graph"]["roc_auc"] = float(roc_auc_score(y_true, test["graph_score"]))
    metrics["graph"]["pr_auc"] = float(average_precision_score(y_true, test["graph_score"]))
    metrics["graph"]["f1"] = float(f1_score(y_true, (test["graph_score"] > g_t).astype(int)))
    metrics["graph"]["f2"] = float(
        fbeta_score_safe(y_true, (test["graph_score"] > g_t).astype(int), beta=2.0)
    )
    metrics["graph"]["accuracy"] = float(
        accuracy_score(y_true, (test["graph_score"] > g_t).astype(int))
    )

    metrics["pairwise"]["roc_auc"] = float(roc_auc_score(y_true, test["pairwise_score"]))
    metrics["pairwise"]["pr_auc"] = float(average_precision_score(y_true, test["pairwise_score"]))
    metrics["pairwise"]["f1"] = float(f1_score(y_true, (test["pairwise_score"] > p_t).astype(int)))
    metrics["pairwise"]["f2"] = float(
        fbeta_score_safe(y_true, (test["pairwise_score"] > p_t).astype(int), beta=2.0)
    )
    metrics["pairwise"]["accuracy"] = float(
        accuracy_score(y_true, (test["pairwise_score"] > p_t).astype(int))
    )

    # Step 11: Save metrics
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_path = f"{cfg.output_dir}/triage_metrics.json"
    plot_model_selection(
        context_cv,
        "Context Model Selection (ROC-AUC vs C)",
        f"{cfg.output_dir}/model_selection_context.png",
    )
    plot_model_selection(
        graph_cv,
        "Graph Model Selection (ROC-AUC vs C)",
        f"{cfg.output_dir}/model_selection_graph.png",
    )
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Step 12: Print summary
    print("=== Classification Report (Context Model) ===")
    print(
        classification_report(
            test["is_critical"],
            (test["context_score"] > ctx_t).astype(int),
        )
    )
    print("=== Classification Report (Graph Model) ===")
    print(
        classification_report(
            test["is_critical"],
            (test["graph_score"] > g_t).astype(int),
        )
    )
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
