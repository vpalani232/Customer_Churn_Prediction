"""SageMaker Training entry point (pipeline step 2).

Fits the chosen model on the training split, picks the decision threshold from 5-fold
out-of-fold predictions (cost based), and saves model.joblib containing the pipeline,
the threshold, and the feature list.
"""
import argparse
import json
import os

import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config, data_utils, features, modeling

TUNABLE = {  # hyperparameter name -> (type, models that use it)
    "learning_rate": (float, {"gb"}), "max_depth": (int, {"gb", "rf"}), "max_iter": (int, {"gb"}),
    "min_samples_leaf": (int, {"gb", "rf"}), "l2_regularization": (float, {"gb"}),
    "n_estimators": (int, {"rf"}), "C": (float, {"logreg"}),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="gb", choices=["logreg", "rf", "gb"])
    ap.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    for name, (typ, _) in TUNABLE.items():
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None)
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    ap.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", ""))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return ap.parse_args()


def main():
    args = parse_args()
    params = {k: getattr(args, k) for k, (_, used_by) in TUNABLE.items()
              if getattr(args, k) is not None and args.model_type in used_by}
    print("model:", args.model_type, "params:", json.dumps(params))

    train = data_utils.read_table(args.train)
    X, y = train[features.MODEL_FEATURES], train["churn"].astype(int)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=config.RANDOM_SEED)
    oof = cross_val_predict(modeling.build_model(args.model_type, **params), X, y, cv=cv,
                            method="predict_proba")[:, 1]
    threshold = modeling.best_threshold(y, oof)
    print(f"cv_roc_auc={modeling.metrics(y, oof, threshold)['roc_auc']:.4f} threshold={threshold}")

    pipeline = modeling.build_model(args.model_type, **params).fit(X, y)
    bundle = {"pipeline": pipeline, "threshold": threshold, "features": features.MODEL_FEATURES,
              "model_type": args.model_type, "params": params, "risk_tiers": config.RISK_TIERS}
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(args.model_dir, "model.joblib"))

    if args.test and os.path.exists(args.test):
        test = data_utils.read_table(args.test)
        for key, value in modeling.metrics(test["churn"], modeling.score_bundle(bundle, test), threshold).items():
            if isinstance(value, float):
                print(f"test_{key}={value:.4f}")


if __name__ == "__main__":
    main()
