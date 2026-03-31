from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from src.utils import load_params

config = load_params()

def train_model(X_train, y_train, model_type):

    if model_type == "Random Forest":

        model_params = config['models']['params_RF']

        model = RandomForestRegressor(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            random_state=model_params['random_state'],
            n_jobs=model_params['n_jobs']
        )
    
    elif model_type == "Linear Regression":

        model = LinearRegression()

    elif model_type == "XGBoost":

        model_params = config['models']['params_XGB']

        model = XGBRegressor(
            n_estimators = model_params['n_estimators'],
            learning_rate = model_params['learning_rate'],
            max_depth = model_params['max_depth'],
            sub_sample = model_params['sub_sample'],
            colsample_bytree = model_params['colsample_bytree']
        )

    else:
        raise ValueError(f"Model type {model_type} not supported.")
    
    model.fit(X_train, y_train)

    return model

def model_prediction(model, X_val):
    y_pred = model.predict(X_val)
    return y_pred