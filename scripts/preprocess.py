"""SageMaker Processing entry point: clean, engineer features, and split (pipeline step 1)."""
import argparse
import os

from sklearn.model_selection import train_test_split

from src import config, data_utils, features, preprocessing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/opt/ml/processing/input/raw")
    ap.add_argument("--output-dir", default="/opt/ml/processing/output")
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    args = ap.parse_args()

    train_raw = data_utils.read_table(os.path.join(args.raw_dir, "train", config.FILES["train"]))
    test_raw = data_utils.read_table(os.path.join(args.raw_dir, "test", config.FILES["test"]))

    train_clean, report = preprocessing.clean(train_raw)
    test_clean, _ = preprocessing.clean(test_raw)
    print("cleaning report:", report)

    train_fe = features.engineer_features(train_clean)
    production = features.engineer_features(test_clean)
    train_df, test_df = train_test_split(train_fe, test_size=config.TEST_SIZE,
                                         stratify=train_fe["churn"], random_state=config.RANDOM_SEED)

    for name, frame in {"train": train_df, "test": test_df, "production": production}.items():
        out_dir = os.path.join(args.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.{args.format}")
        frame.to_parquet(path, index=False) if args.format == "parquet" else frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
