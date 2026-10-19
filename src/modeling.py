"""Model building, threshold selection, and evaluation helpers shared by notebooks and scripts."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, features

AGE_BANDS = ([17, 29, 49, 65], ["18-29", "30-49", "50-65"])


# ----------------------------------------------------------------------------- pipelines
def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Impute, scale (linear model only), and one-hot encode. Fitted on training data only."""
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric), features.NUMERIC_FEATURES),
        ("bin", "passthrough", features.BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features.CATEGORICAL_FEATURES),
    ])


def build_model(model_type: str, **params) -> Pipeline:
    """Return an unfitted pipeline (preprocessing + estimator). model_type: logreg | rf | gb."""
    seed = config.RANDOM_SEED
    if model_type == "logreg":
        clf = LogisticRegression(**{"max_iter": 1000, "class_weight": "balanced", **params})
    elif model_type == "rf":
        clf = RandomForestClassifier(**{"n_estimators": 200, "min_samples_leaf": 5, "n_jobs": -1,
                                        "class_weight": "balanced", "random_state": seed, **params})
    elif model_type == "gb":
        clf = HistGradientBoostingClassifier(**{"max_iter": 300, "learning_rate": 0.1,
                                                "random_state": seed, **params})
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    return Pipeline([("prep", build_preprocessor(scale=model_type == "logreg")), ("clf", clf)])


def benchmark_heuristic(df: pd.DataFrame) -> np.ndarray:
    """Simple rule-based benchmark: the share of four risk rules a customer triggers (0 to 1)."""
    rules = [df["support_calls"] >= features.MANY_SUPPORT_CALLS,
             df["payment_delay"] > features.HIGH_PAYMENT_DELAY_DAYS,
             df["monthly_contract"] == 1,
             df["tenure"] <= 12]
    return np.mean([r.astype(float).to_numpy() for r in rules], axis=0)


def benchmark_two_feature_model() -> Pipeline:
    """Benchmark model with just two features (support calls and payment delay)."""
    prep = ColumnTransformer([("num", StandardScaler(), ["support_calls", "payment_delay"])])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=500))])


# ----------------------------------------------------------------------------- evaluation
def expected_cost(y, pred, cost_fn=config.COST_FALSE_NEGATIVE, cost_fp=config.COST_FALSE_POSITIVE) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return float(fn * cost_fn + fp * cost_fp)


def best_threshold(y, proba, cost_fn=config.COST_FALSE_NEGATIVE, cost_fp=config.COST_FALSE_POSITIVE) -> float:
    """Threshold that minimizes expected cost (a missed churner costs more than a wasted offer)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [expected_cost(y, (proba >= t).astype(int), cost_fn, cost_fp) for t in grid]
    return float(grid[int(np.argmin(costs))])


def top_decile_precision(y, proba) -> float:
    y, proba = np.asarray(y), np.asarray(proba)
    k = max(1, int(len(y) * 0.10))
    return float(y[np.argsort(-proba)[:k]].mean())


def lift_table(y, proba, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba)}).sort_values("p", ascending=False)
    df["decile"] = np.concatenate([[i + 1] * len(c) for i, c in enumerate(np.array_split(np.arange(len(df)), bins))])
    out = df.groupby("decile")["y"].agg(["mean", "count"]).rename(columns={"mean": "churn_rate"})
    out["lift"] = out["churn_rate"] / df["y"].mean()
    return out


def metrics(y, proba, threshold: float) -> dict:
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "top_decile_precision": top_decile_precision(y, proba),
        "expected_cost": expected_cost(y, pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(threshold),
    }


def fairness_table(df: pd.DataFrame, pred, group_col: str) -> pd.DataFrame:
    """Selection rate, recall, and false-positive rate per group (gender or age band)."""
    d = df[[group_col, "churn"]].copy()
    d.columns = ["group", "y"]
    d["pred"] = np.asarray(pred)
    rows = []
    for g, part in d.groupby("group", observed=True):
        pos, neg = part[part.y == 1], part[part.y == 0]
        rows.append({"group": g, "n": len(part), "selection_rate": part.pred.mean(),
                     "recall": pos.pred.mean() if len(pos) else np.nan,
                     "false_positive_rate": neg.pred.mean() if len(neg) else np.nan})
    return pd.DataFrame(rows).set_index("group")


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_band"] = pd.cut(out["age"], bins=AGE_BANDS[0], labels=AGE_BANDS[1]).astype(str)
    return out


def max_fairness_gap(table: pd.DataFrame) -> float:
    """Largest between-group gap across selection rate, recall, and false-positive rate."""
    gaps = [table[c].max() - table[c].min() for c in ["selection_rate", "recall", "false_positive_rate"]]
    return float(np.nanmax(gaps))


def evaluation_report(df: pd.DataFrame, proba, threshold: float) -> dict:
    """JSON-ready report used by the notebooks, the pipeline gate, and the model registry."""
    m = metrics(df["churn"], proba, threshold)
    pred = (np.asarray(proba) >= threshold).astype(int)
    d = add_age_band(df)
    gaps = {col: max_fairness_gap(fairness_table(d, pred, col)) for col in ("gender", "age_band")}
    return {"metrics": {k: {"value": v} for k, v in m.items()},
            "fairness_max_gap": {k: float(v) for k, v in gaps.items()}}


def score_bundle(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Probability of churn from a saved model bundle (pipeline + threshold + feature list)."""
    return bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]
