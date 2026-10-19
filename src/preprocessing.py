"""Deterministic cleaning rules for the raw churn data.

The same function is used by the training notebooks and (later) the serving code,
so training and inference see identically prepared data.
"""
import pandas as pd

from . import config

# Expected value ranges (inclusive) from the dataset description; rows outside are dropped.
VALID_RANGES = {
    "Age": (18, 65),
    "Tenure": (1, 60),
    "Usage Frequency": (1, 30),
    "Support Calls": (0, 10),
    "Payment Delay": (0, 30),
    "Last Interaction": (1, 30),
}
CATEGORICAL = ["Gender", "Subscription Type", "Contract Length"]
EXPECTED_COLUMNS = (
    [config.ID_COLUMN, "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
     "Payment Delay", "Subscription Type", "Contract Length", "Total Spend",
     "Last Interaction", config.TARGET]
)
# If fewer than this share of rows contain a null feature value, drop them; otherwise impute.
MAX_DROP_FRACTION = 0.01


def clean(df: pd.DataFrame, max_drop_fraction: float = MAX_DROP_FRACTION):
    """Return (clean_df, report). `report` records how many rows each rule removed."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[EXPECTED_COLUMNS].copy()
    report = {"rows_in": len(df)}

    # 1. Rows without a label cannot be used for supervised learning
    df = df.dropna(subset=[config.TARGET, config.ID_COLUMN])
    report["dropped_missing_label_or_id"] = report["rows_in"] - len(df)

    # 2. Exact duplicates and duplicate customers
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset=[config.ID_COLUMN], keep="first")
    report["dropped_duplicates"] = before - len(df)

    # 3. Nulls in feature columns: drop if rare, otherwise impute (median / mode)
    feature_cols = [c for c in EXPECTED_COLUMNS if c not in (config.TARGET, config.ID_COLUMN)]
    null_rows = df[feature_cols].isna().any(axis=1)
    report["rows_with_null_features"] = int(null_rows.sum())
    if null_rows.mean() < max_drop_fraction:
        df = df[~null_rows]
        report["null_rule"] = "dropped"
    else:
        for col in feature_cols:
            if col in CATEGORICAL:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        report["null_rule"] = "imputed"

    # 4. Standardize categorical text, then range-check numeric columns
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip().str.title()
    before = len(df)
    for col, (lo, hi) in VALID_RANGES.items():
        df = df[df[col].between(lo, hi)]
    df = df[df["Total Spend"] > 0]
    report["dropped_out_of_range"] = before - len(df)

    # 5. Types
    df[config.TARGET] = df[config.TARGET].astype(int)
    df[config.ID_COLUMN] = df[config.ID_COLUMN].astype(int)
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report
