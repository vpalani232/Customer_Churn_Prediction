"""Data loading helpers shared across notebooks."""
import os

import pandas as pd

from . import config


def load_raw(split: str) -> pd.DataFrame:
    """Load the raw CSV for `split` ('train' or 'test').

    Reads from S3 by default. Set LOCAL_DATA_DIR to read from a local folder
    instead (useful for quick tests outside AWS).
    """
    if split not in config.FILES:
        raise ValueError(f"split must be one of {list(config.FILES)}")
    filename = config.FILES[split]

    local_dir = os.getenv("LOCAL_DATA_DIR")
    if local_dir:
        return pd.read_csv(os.path.join(local_dir, filename))

    import awswrangler as wr

    path = f"s3://{config.get_bucket()}/{config.PREFIXES[split]}/{filename}"
    return wr.s3.read_csv(path)


def read_table(path: str) -> pd.DataFrame:
    """Read a parquet or CSV file, or the first such file found inside a directory."""
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path) if f.endswith((".parquet", ".csv")))
        if not names:
            raise FileNotFoundError(f"no .parquet or .csv file in {path}")
        path = os.path.join(path, names[0])
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def load_processed(name: str) -> pd.DataFrame:
    """Load a processed split ('train', 'test' or 'production') from S3.

    Set LOCAL_PROCESSED_DIR to read <dir>/<name>.parquet|csv from disk instead.
    """
    local_dir = os.getenv("LOCAL_PROCESSED_DIR")
    if local_dir:
        for candidate in (f"{name}.parquet", f"{name}.csv", name):
            path = os.path.join(local_dir, candidate)
            if os.path.exists(path):
                return read_table(path)
        raise FileNotFoundError(f"no {name} data in {local_dir}")
    import awswrangler as wr

    return wr.s3.read_parquet(f"s3://{config.get_bucket()}/{config.PROCESSED_PREFIX}/{name}/{name}.parquet")
