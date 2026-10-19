"""Feature engineering. Every feature is computed row by row, so nothing here is fitted
on data and the split can safely happen after it (no leakage).

Scaling and one-hot encoding are handled later inside the scikit-learn training pipeline.
"""
import re

import numpy as np
import pandas as pd


HIGH_PAYMENT_DELAY_DAYS = 15      # confirm against the EDA churn-rate-by-delay chart
MANY_SUPPORT_CALLS = 5            # confirm against the EDA churn-rate-by-calls chart
TENURE_BINS = [0, 12, 24, 36, 60]
TENURE_LABELS = ["0-12", "13-24", "25-36", "37-60"]

# Columns kept out of the model: identifier, label, and Gender (excluded for fairness;
# it stays in the data only so the bias audit can compare groups).
EXCLUDED_FROM_MODEL = ["customer_id", "churn", "gender"]

PROFILE_FEATURES = ["age", "tenure", "subscription_type", "contract_length",
                    "tenure_bucket", "monthly_contract"]
ACTIVITY_FEATURES = ["usage_frequency", "support_calls", "payment_delay", "last_interaction",
                     "total_spend", "spend_per_month", "calls_per_month", "engagement_ratio",
                     "high_payment_delay", "many_support_calls"]
NUMERIC_FEATURES = ["age", "tenure", "usage_frequency", "support_calls", "payment_delay",
                    "last_interaction", "total_spend", "spend_per_month", "calls_per_month",
                    "engagement_ratio"]
CATEGORICAL_FEATURES = ["subscription_type", "contract_length", "tenure_bucket"]
BINARY_FEATURES = ["monthly_contract", "high_payment_delay", "many_support_calls"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def to_snake(name: str) -> str:
    """'Usage Frequency' -> 'usage_frequency', 'CustomerID' -> 'customer_id'."""
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name.strip())
    return re.sub(r"\s+", "_", name).lower()


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Take the output of preprocessing.clean() and return snake_case columns plus new features."""
    df = clean_df.rename(columns={c: to_snake(c) for c in clean_df.columns}).copy()

    tenure = df["tenure"].clip(lower=1)                    # guard against division by zero
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    df["spend_per_month"] = df["total_spend"] / tenure
    df["calls_per_month"] = df["support_calls"] / tenure
    df["engagement_ratio"] = df["usage_frequency"] / df["last_interaction"].clip(lower=1)
    df["high_payment_delay"] = (df["payment_delay"] > HIGH_PAYMENT_DELAY_DAYS).astype(int)
    df["many_support_calls"] = (df["support_calls"] >= MANY_SUPPORT_CALLS).astype(int)
    df["monthly_contract"] = (df["contract_length"] == "Monthly").astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def to_feature_store_frame(df: pd.DataFrame, columns, event_time: float) -> pd.DataFrame:
    """Prepare columns for a SageMaker Feature Group (string dtype, id and event time added)."""
    out = df[["customer_id"] + list(columns)].copy()
    out["customer_id"] = out["customer_id"].astype(str)
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("string")   # Feature Store needs the string dtype
    out["event_time"] = float(event_time)
    return out
