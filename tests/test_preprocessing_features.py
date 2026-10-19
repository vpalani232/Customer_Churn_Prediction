import pandas as pd
import pytest

from src import config, features, preprocessing


def make_df(n=200):
    return pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Age": [30 + i % 30 for i in range(n)],
        "Gender": ["Male", "Female"] * (n // 2),
        "Tenure": [1 + i % 60 for i in range(n)],
        "Usage Frequency": [1 + i % 30 for i in range(n)],
        "Support Calls": [i % 11 for i in range(n)],
        "Payment Delay": [i % 31 for i in range(n)],
        "Subscription Type": ["Basic", "Standard", "Premium", "Basic"] * (n // 4),
        "Contract Length": ["Monthly", "Annual", "Quarterly", "Monthly"] * (n // 4),
        "Total Spend": [100.0 + i for i in range(n)],
        "Last Interaction": [1 + i % 30 for i in range(n)],
        "Churn": [i % 2 for i in range(n)],
    })


def test_clean_keeps_good_rows_and_reports():
    out, report = preprocessing.clean(make_df())
    assert len(out) == 200 and report["rows_out"] == 200
    assert out[config.TARGET].isin([0, 1]).all()


def test_clean_drops_duplicates_missing_label_and_out_of_range():
    df = make_df()
    df = pd.concat([df, df.iloc[[0]]])             # exact duplicate
    df.loc[1, "Churn"] = None                       # missing label
    df.loc[2, "Age"] = 99                           # out of range
    out, report = preprocessing.clean(df)
    assert report["dropped_missing_label_or_id"] == 1
    assert report["dropped_duplicates"] == 1
    assert report["dropped_out_of_range"] == 1
    assert len(out) == 198


def test_clean_requires_expected_columns():
    with pytest.raises(ValueError):
        preprocessing.clean(make_df().drop(columns=["Age"]))


def test_engineer_features_columns_and_no_infinities():
    out = features.engineer_features(preprocessing.clean(make_df())[0])
    for col in features.MODEL_FEATURES + ["customer_id", "churn", "gender"]:
        assert col in out.columns
    assert not out[features.MODEL_FEATURES].isin([float("inf")]).any().any()
    assert set(out["tenure_bucket"].unique()) <= set(features.TENURE_LABELS)


def test_gender_and_id_not_model_features():
    assert "gender" not in features.MODEL_FEATURES
    assert "customer_id" not in features.MODEL_FEATURES


def test_feature_store_frame_has_id_and_event_time():
    out = features.engineer_features(preprocessing.clean(make_df())[0])
    fs = features.to_feature_store_frame(out, features.PROFILE_FEATURES, 123.0)
    assert {"customer_id", "event_time"} <= set(fs.columns)
    assert str(fs["customer_id"].dtype) in ("object", "string", "str")
