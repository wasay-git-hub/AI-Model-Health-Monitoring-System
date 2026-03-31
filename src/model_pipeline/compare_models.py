import pandas as pd
import numpy as np
import joblib
import os
from scipy import stats
from src.utils import load_params
from src.model_pipeline.preprocessing import load_and_merge, clean_data, feature_engineering, split_data

def run_model_comparison():
    print("Running Statistical Model Comparison (ANOVA)")

    # Load configuration
    config = load_params()
    train_path = config['paths']['train_dataset']
    store_path = config['paths']['store_dataset']

    print("Preparing test data")
    df = load_and_merge(train_path, store_path)
    df = clean_data(df)
    df = feature_engineering(df)
    
    # We only need the Test set for this comparison
    _, _, _, _, X_test, y_test = split_data(df)

    # Define models to compare
    model_files = {
        "Linear Regression": "models/LinearRegression.pkl",
        "Random Forest": "models/RandomForest.pkl",
        "XGBoost": "models/XGBoost.pkl"
    }

    squared_errors = {}

    print("Loading models and generating predictions")
    
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"Processing {name}")
            model = joblib.load(path)
            
            # Predict
            preds = model.predict(X_test)
            
            # Calculate Squared Errors for ANOVA
            squared_errors[name] = (y_test - preds) ** 2
        else:
            print(f"Warning: {name} not found at {path}. Skipping.")

    # Ensure we have enough models to compare
    if len(squared_errors) < 2:
        print("Error: Need at least 2 models to perform ANOVA.")
        return

    # Perform One-Way ANOVA
    print("Calculating One-Way ANOVA")
    
    # Extract the error arrays
    error_arrays = list(squared_errors.values())
    
    # Compare the means of the errors
    f_stat, p_value = stats.f_oneway(*error_arrays)

    print("Results:")
    print(f"Comparing: {', '.join(squared_errors.keys())}")
    print(f"F-Statistic: {f_stat:.4f}")
    print(f"P-Value: {p_value:.4e}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("Conclusion: The difference is statistically significant.")
        print("The models perform differently.")
        
        print("Mean Squared Errors (Lower is better):")
        for name, errors in squared_errors.items():
            mse = np.mean(errors)
            print(f"{name}: {mse:.2f}")
            
    else:
        print("Conclusion: The difference is not statistically significant (p > 0.05).")
        print("All models perform roughly the same.")

if __name__ == "__main__":
    run_model_comparison()