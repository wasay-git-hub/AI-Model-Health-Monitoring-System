import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score

def get_rmspe(y_true, y_pred):
    # to avoid division by zero
    non_zero = y_true != 0
    rmspe = np.sqrt(np.mean(((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]) ** 2))
    rmspe = round(rmspe,2)
    return rmspe

def get_evaluations(y_true, y_pred):
    """
    Evaluating through multiple metrics. But I have categorized them based on their usefulness.
    Primary metric: RMSPE
    Secondary metrics: MAPE, MAE, RMSE
    Statistical metric: R2
    """
    # MAE
    mae = round(mean_absolute_error(y_true, y_pred),2)
    # RMSE
    rmse = round(root_mean_squared_error(y_true, y_pred),2)
    # MAPE
    mape = round(mean_absolute_percentage_error(y_true, y_pred),2)
    # RMSPE
    rmspe = get_rmspe(y_true,y_pred)
    # R2
    r2 = r2_score(y_true, y_pred)

    # Primary Metric
    print("\n[ PRIMARY METRIC ]")
    print(f"Root Mean Square Percentage Error (RMSPE): {rmspe}")

    # Secondary Metrics
    print("\n[ SECONDARY METRICS ]")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Statistical Metric
    print("\n[ STATISTICAL METRIC ]")
    print(f"R-Squared (R2 Score): {r2}")

    # returning all as a dictionary
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "RMSPE": rmspe,
        "R2": r2
    }