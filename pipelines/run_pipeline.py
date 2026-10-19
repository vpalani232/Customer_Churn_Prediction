"""Create or update the pipeline and run it. Used by the CD workflow and notebook 07.

Exit code is non-zero when the execution fails, so a failed quality gate fails the CI job.
"""
import argparse
import json
import os
import sys

import boto3
import sagemaker

from pipelines.pipeline import get_pipeline
from src import config


def load_best_params(bucket: str) -> dict:
    """Tuned model settings saved by notebook 04 (empty dict if the notebook has not been run)."""
    try:
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=f"{config.MODEL_PREFIX}/best_params.json")
        return json.loads(obj["Body"].read())
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-roc-auc", type=float, default=config.MIN_ROC_AUC)
    ap.add_argument("--min-recall", type=float, default=config.MIN_RECALL)
    ap.add_argument("--model-type", default=None, help="logreg | rf | gb (default: tuned model from notebook 04)")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    role = os.getenv("SAGEMAKER_ROLE_ARN") or sagemaker.get_execution_role()
    bucket = config.get_bucket()
    best = load_best_params(bucket)
    model_type = args.model_type or best.get("model_type", "gb")
    pipeline = get_pipeline(role=role, bucket=bucket, hyperparameters=best.get("hyperparameters", {}))
    pipeline.upsert(role_arn=role)
    execution = pipeline.start(parameters={"MinRocAuc": args.min_roc_auc, "MinRecall": args.min_recall,
                                           "ModelType": model_type})
    print("started:", execution.arn)
    if args.no_wait:
        return 0
    try:
        execution.wait(delay=30, max_attempts=120)
    except Exception as err:                      # a FailStep makes wait() raise
        print("pipeline did not succeed:", err)
        status = boto3.client("sagemaker", region_name=config.REGION).describe_pipeline_execution(
            PipelineExecutionArn=execution.arn)["PipelineExecutionStatus"]
        print("status:", status)
        return 1
    print("pipeline succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
