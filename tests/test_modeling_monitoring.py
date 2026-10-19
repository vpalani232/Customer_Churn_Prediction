import numpy as np
import pandas as pd

from src import features, modeling, monitoring, preprocessing
from tests.test_preprocessing_features import make_df


def make_features(n=800, seed=0):
    rng = np.random.default_rng(seed)
    df = make_df(n)
    df["Churn"] = (rng.random(n) < 0.3 + 0.05 * df["Support Calls"]).astype(int)
    return features.engineer_features(preprocessing.clean(df)[0])


def test_best_threshold_prefers_catching_churners_when_misses_cost_more():
    y = np.array([1] * 50 + [0] * 50)
    proba = np.concatenate([np.linspace(0.3, 0.9, 50), np.linspace(0.05, 0.6, 50)])
    t_default = modeling.best_threshold(y, proba, cost_fn=20, cost_fp=20)
    t_costly_miss = modeling.best_threshold(y, proba, cost_fn=200, cost_fp=20)
    assert t_costly_miss <= t_default


def test_metrics_and_top_decile():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = modeling.metrics(y, proba, 0.5)
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 0 and m["tn"] == 6
    assert modeling.top_decile_precision(y, proba) == 1.0


def test_all_models_train_and_predict():
    df = make_features()
    X, y = df[features.MODEL_FEATURES], df["churn"]
    for name in ["logreg", "rf", "gb"]:
        model = modeling.build_model(name, **({"n_estimators": 20} if name == "rf" else {})).fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()


def test_gender_never_reaches_the_model():
    assert "gender" not in features.MODEL_FEATURES


def test_benchmark_heuristic_range():
    df = make_features()
    score = modeling.benchmark_heuristic(df)
    assert score.min() >= 0 and score.max() <= 1


def test_evaluation_report_has_gate_metrics_and_fairness():
    df = make_features()
    report = modeling.evaluation_report(df, np.random.default_rng(1).random(len(df)), 0.5)
    assert {"roc_auc", "recall"} <= set(report["metrics"])
    assert set(report["fairness_max_gap"]) == {"gender", "age_band"}


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert monitoring.psi_numeric(ref, rng.normal(0, 1, 5000)) < 0.05
    assert monitoring.psi_numeric(ref, rng.normal(2, 1, 5000)) > 0.2
    assert monitoring.psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 95 + ["b"] * 5) > 0.2
