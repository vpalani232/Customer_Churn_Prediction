"""Data drift helpers (Population Stability Index) used by the monitoring notebook and tests."""
import numpy as np
import pandas as pd

PSI_ALERT = 0.20          # common rule of thumb: > 0.2 indicates a significant shift
_EPS = 1e-6


def psi_numeric(reference, live, bins: int = 10) -> float:
    """PSI of a numeric feature, using quantile bins of the reference (training) data."""
    reference = pd.Series(reference).dropna().to_numpy()
    live = pd.Series(live).dropna().to_numpy()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, edges)[0] / len(reference) + _EPS
    live_p = np.histogram(live, edges)[0] / len(live) + _EPS
    return float(np.sum((live_p - ref_p) * np.log(live_p / ref_p)))


def psi_categorical(reference, live) -> float:
    ref = pd.Series(reference).value_counts(normalize=True)
    liv = pd.Series(live).value_counts(normalize=True)
    cats = ref.index.union(liv.index)
    ref_p = ref.reindex(cats, fill_value=0).to_numpy() + _EPS
    live_p = liv.reindex(cats, fill_value=0).to_numpy() + _EPS
    return float(np.sum((live_p - ref_p) * np.log(live_p / ref_p)))


def drift_report(reference: pd.DataFrame, live: pd.DataFrame, numeric, categorical,
                 threshold: float = PSI_ALERT) -> pd.DataFrame:
    """One row per feature with its PSI and a drift flag, sorted from most to least drifted."""
    rows = [{"feature": c, "psi": psi_numeric(reference[c], live[c])} for c in numeric]
    rows += [{"feature": c, "psi": psi_categorical(reference[c], live[c])} for c in categorical]
    out = pd.DataFrame(rows)
    out["drifted"] = out["psi"] > threshold
    return out.sort_values("psi", ascending=False).reset_index(drop=True)
