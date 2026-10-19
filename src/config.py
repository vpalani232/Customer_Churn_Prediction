"""Central project configuration used by every notebook and script."""
import os

REGION = os.getenv("AWS_REGION", "us-east-1")

# S3 layout (all prefixes live inside one bucket)
RAW_PREFIX = "churn/raw"
PREFIXES = {
    "train": f"{RAW_PREFIX}/train",   # Kaggle training file -> Athena table churn_train
    "test": f"{RAW_PREFIX}/test",     # Kaggle testing file  -> Athena table churn_test (held-out "production" data)
}
PROCESSED_PREFIX = "churn/processed"
ATHENA_RESULTS_PREFIX = "churn/athena-results"

# Kaggle "Customer Churn Dataset" (muhammadshahidazeem) file names.
# Adjust here if your downloaded files are named differently.
FILES = {
    "train": "customer_churn_dataset-training-master.csv",
    "test": "customer_churn_dataset-testing-master.csv",
}

GLUE_DATABASE = "customer_churn_db"
TABLES = {"train": "churn_train", "test": "churn_test"}

TARGET = "Churn"
ID_COLUMN = "CustomerID"
RANDOM_SEED = 42
TEST_SIZE = 0.20          # 80/20 train/test split of the training file

# Course requirement: >10,000 labeled records per class and >10,000 rows per file
MIN_ROWS_PER_FILE = 10_000
MIN_ROWS_PER_CLASS = 10_000


def get_bucket() -> str:
    """Return the S3 bucket: CHURN_BUCKET env var, else the SageMaker default bucket."""
    bucket = os.getenv("CHURN_BUCKET")
    if bucket:
        return bucket
    import sagemaker  # imported lazily so the module also works outside SageMaker

    return sagemaker.Session().default_bucket()


# ---- Modeling and deployment settings -------------------------------------------------
SELECTION_SAMPLE_ROWS = 100_000   # stratified sample used for model comparison and tuning
CV_FOLDS = 5

# Business costs used to pick the decision threshold (placeholder assumptions: replace with
# the company's real numbers). A missed churner loses the customer's margin; a false alarm
# costs one retention offer.
COST_FALSE_NEGATIVE = 100.0
COST_FALSE_POSITIVE = 20.0

RISK_TIERS = (0.40, 0.70)         # probability cutoffs: low < 0.40 <= medium < 0.70 <= high

# Quality gate used by the CI/CD pipeline (the Design Document goals; adjust after baselines)
MIN_ROC_AUC = 0.85
MIN_RECALL = 0.80
MAX_FAIRNESS_GAP = 0.10

MODEL_PREFIX = "churn/models"
MONITOR_PREFIX = "churn/monitoring"
MODEL_PACKAGE_GROUP = "churn-model-group"
ENDPOINT_NAME = "churn-endpoint"
PIPELINE_NAME = "churn-training-pipeline"
