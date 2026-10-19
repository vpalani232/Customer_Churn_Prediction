"""SageMaker inference handlers (real-time endpoint and batch transform).

Input: CSV rows (no header) in the order of the model feature list, or JSON
({"instances": [{...}]} or a list of records).
Output: CSV lines 'probability,prediction,risk_tier' or JSON.
"""
import json
import os
from io import StringIO

import joblib
import pandas as pd


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(body, content_type="text/csv"):
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    if content_type == "text/csv":
        return pd.read_csv(StringIO(body), header=None)      # columns are named in predict_fn
    if content_type == "application/json":
        payload = json.loads(body)
        records = payload["instances"] if isinstance(payload, dict) else payload
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported content type: {content_type}")


def _tier(p, tiers):
    return "low" if p < tiers[0] else "medium" if p < tiers[1] else "high"


def predict_fn(df, bundle):
    feats = bundle["features"]
    if list(df.columns) != feats:
        if df.shape[1] != len(feats):
            raise ValueError(f"expected {len(feats)} columns, got {df.shape[1]}")
        df.columns = feats
    proba = bundle["pipeline"].predict_proba(df[feats])[:, 1]
    return pd.DataFrame({"probability": proba,
                         "prediction": (proba >= bundle["threshold"]).astype(int),
                         "risk_tier": [_tier(p, bundle["risk_tiers"]) for p in proba]})


def output_fn(result, accept="text/csv"):
    if accept == "application/json":
        return json.dumps(result.to_dict(orient="records")), accept
    lines = [f"{r.probability:.6f},{int(r.prediction)},{r.risk_tier}" for r in result.itertuples()]
    return "\n".join(lines), "text/csv"
