import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing import load_and_merge, process_data, split_raw_data
from src.utils import load_params


def _prepare_train_stats(train_dataset, store_dataset):
    """
    Build preprocessing stats from the training split only so external input
    files are transformed exactly like CV/Test data.
    """
    full_df = load_and_merge(train_dataset, store_dataset)
    train_raw, _, _ = split_raw_data(full_df)
    _, train_stats = process_data(train_raw, train_stats=None)
    return train_stats


def _load_input_with_store(input_csv_path, store_dataset):
    """
    If store metadata columns are missing, merge input rows with store.csv.
    If input is already merged, keep it as-is.
    """
    df = pd.read_csv(input_csv_path, low_memory=False)
    required_store_cols = {"StoreType", "Assortment", "CompetitionDistance"}

    if required_store_cols.issubset(df.columns):
        return df

    store_df = pd.read_csv(store_dataset)
    return pd.merge(df, store_df, how="left", on="Store")


def preprocess_input_file(input_csv_path, output_csv_path=None, use_train_stats=False, features_only=True):
    config = load_params()
    train_dataset = config["paths"]["train_dataset"]
    store_dataset = config["paths"]["store_dataset"]

    raw_input_df = _load_input_with_store(input_csv_path, store_dataset)

    # Default behavior avoids using train-derived statistics.
    # If explicitly requested, we mimic CV/Test preprocessing with train stats.
    train_stats = None
    if use_train_stats:
        train_stats = _prepare_train_stats(train_dataset, store_dataset)

    processed_df, _ = process_data(raw_input_df, train_stats=train_stats)

    if features_only:
        feature_cols = config["training_data"]["features"]
        processed_df = processed_df[feature_cols]

    input_path = Path(input_csv_path)
    if output_csv_path is None:
        output_csv_path = str(input_path.with_name(f"processed_{input_path.name}"))

    processed_df.to_csv(output_csv_path, index=False)
    return output_csv_path, len(processed_df)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess one or more input CSV files using train-derived preprocessing stats."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input CSV paths (e.g., data/input_1.csv data/input_2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. By default, output is written next to each input file.",
    )
    parser.add_argument(
        "--use-train-stats",
        action="store_true",
        help="Use train-derived preprocessing stats (same behavior as CV/Test preprocessing).",
    )
    parser.add_argument(
        "--keep-all-columns",
        action="store_true",
        help="Keep all processed columns instead of outputting only model feature columns.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        input_file = Path(input_path)
        output_path = None
        if output_dir is not None:
            output_path = str(output_dir / f"processed_{input_file.name}")

        saved_to, rows = preprocess_input_file(
            input_path,
            output_path,
            use_train_stats=args.use_train_stats,
            features_only=not args.keep_all_columns,
        )
        print(f"Processed {input_path} -> {saved_to} ({rows} rows)")


if __name__ == "__main__":
    main()