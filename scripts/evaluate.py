"""SageMaker Processing entry point: evaluate a trained model on the test split (pipeline step 3).

Writes evaluation.json, which the pipeline quality gate and the model registry read.
"""
import argparse
import json
import os
import tarfile

import joblib

from src import data_utils, modeling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/opt/ml/processing/model")
    ap.add_argument("--test-dir", default="/opt/ml/processing/test")
    ap.add_argument("--output-dir", default="/opt/ml/processing/evaluation")
    args = ap.parse_args()

    archive = os.path.join(args.model_dir, "model.tar.gz")
    if os.path.exists(archive):
        with tarfile.open(archive) as tar:
            tar.extractall(args.model_dir)
    bundle = joblib.load(os.path.join(args.model_dir, "model.joblib"))

    test = data_utils.read_table(args.test_dir)
    report = modeling.evaluation_report(test, modeling.score_bundle(bundle, test), bundle["threshold"])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v["value"] for k, v in report["metrics"].items() if k in ("roc_auc", "recall", "precision")}))


if __name__ == "__main__":
    main()
