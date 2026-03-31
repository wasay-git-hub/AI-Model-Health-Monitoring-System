from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from src.model_pipeline.evaluation import get_rmspe
from src.model_pipeline.preprocessing import process_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def _resolve_input_path(input_file):
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path
    return input_path.resolve()

def _load_input_with_store(input_path, store_dataset_path):
    raw_df = pd.read_csv(input_path, low_memory=False)

    store_cols = {"StoreType", "Assortment", "CompetitionDistance"}
    if store_cols.issubset(raw_df.columns):
        return raw_df

    store_df = pd.read_csv(store_dataset_path)
    return pd.merge(raw_df, store_df, how="left", on="Store")

def _compute_metrics(y_true, y_pred):
    non_zero = y_true != 0
    y_true_safe = y_true[non_zero]
    y_pred_safe = y_pred[non_zero]

    mae = float(round(mean_absolute_error(y_true, y_pred), 4))
    rmse = float(round(root_mean_squared_error(y_true, y_pred), 4))
    mape = float(round(mean_absolute_percentage_error(y_true_safe, y_pred_safe), 4))
    rmspe = float(get_rmspe(y_true, y_pred))
    r2 = float(round(r2_score(y_true, y_pred), 4))

    return {
        "RMSPE": rmspe,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }

def evaluate_input_file(input_file, config, model):
    input_path = _resolve_input_path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    store_dataset_path = BASE_DIR / config["paths"]["store_dataset"]
    merged_df = _load_input_with_store(input_path, store_dataset_path)

    target_col = config["training_data"]["target"]
    feature_cols = config["training_data"]["features"]
    if target_col not in merged_df.columns:
        raise ValueError(
            f"Input file must contain target column '{target_col}' to compute metrics."
        )

    processed_df, _ = process_data(merged_df, train_stats=None)
    if processed_df.empty:
        raise ValueError("Input file has no valid rows after preprocessing.")

    missing_features = [col for col in feature_cols if col not in processed_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features after preprocessing: {missing_features}")

    X = processed_df[feature_cols]
    y_true = processed_df[target_col]
    y_pred = model.predict(X)

    metrics = _compute_metrics(y_true, y_pred)
    prediction_summary = {
        "min_prediction": float(np.min(y_pred)),
        "max_prediction": float(np.max(y_pred)),
        "mean_prediction": float(np.mean(y_pred)),
    }

    return {
        "input_file": str(input_path),
        "row_count": int(len(processed_df)),
        "metrics": metrics,
        "prediction_summary": prediction_summary,
    }

def compare_input_files(input_files, config, model):
    results = [evaluate_input_file(path, config, model) for path in input_files]

    metric_directions = {
        "RMSPE": "min",
        "MAE": "min",
        "RMSE": "min",
        "MAPE": "min",
        "R2": "max",
    }

    rankings = []
    for metric_name, direction in metric_directions.items():
        ordered = sorted(
            results,
            key=lambda item: item["metrics"][metric_name],
            reverse=(direction == "max"),
        )
        rankings.append(
            {
                "metric": metric_name,
                "best_file": ordered[0]["input_file"],
                "worst_file": ordered[-1]["input_file"],
            }
        )

    return {
        "results": results,
        "rankings": rankings,
    }