import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from pathlib import Path
from src.utils import load_params
from src.model_pipeline.preprocessing import load_and_merge, split_raw_data, process_data, extract_X_y
from src.model_pipeline.evaluation import get_evaluations

def log_existing_models():
    config = load_params()
    train_dataset = config['paths']['train_dataset']
    store_dataset = config['paths']['store_dataset']

    # Setup MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Model_Training")

    # Prepare Data
    print("Loading data for metadata logging...")
    df = load_and_merge(train_dataset, store_dataset)
    train_raw, cv_raw, test_raw = split_raw_data(df)
    train_df, train_stats = process_data(train_raw, train_stats=None)
    test_df, _ = process_data(test_raw, train_stats=train_stats)
    X_test, y_test = extract_X_y(test_df)

    model_names = ["XGBoost"]

    for name in model_names:
        model_path = Path(f"models/{name}.pkl")
        
        if model_path.exists():
            print(f"Logging metadata for: {name}")
            model = joblib.load(model_path)
            
            # Quick evaluation
            preds = model.predict(X_test)
            metrics = get_evaluations(y_test, preds)

            with mlflow.start_run(run_name=f"Baseline_{name}"):
                # Log the parameters from your yaml
                mlflow.log_param("model_type", name)
                
                # Log the metrics
                for m_name, m_value in metrics.items():
                    mlflow.log_metric(f"test_{m_name}", m_value)
                
                # Log the model itself as an artifact
                mlflow.sklearn.log_model(model, "model_artifact")
        else:
            print(f"Skipping {name}: file not found at {model_path}")

if __name__ == "__main__":
    log_existing_models()