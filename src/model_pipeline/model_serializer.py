import joblib
from pathlib import Path
from src.utils import load_params

config = load_params()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def save_model(model, model_type):
    if model_type == "Random Forest":
        MODEL_PATH = BASE_DIR/config['paths']['model_1']
    
    elif model_type == "Linear Regression":
        MODEL_PATH = BASE_DIR/config['paths']['model_2']

    elif model_type == "XGBoost":
        MODEL_PATH = BASE_DIR/config['paths']['model_3']

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)

def load_model(model_type):
    if model_type == "Random Forest":
        MODEL_PATH = BASE_DIR/config['paths']['model_1']
    
    elif model_type == "Linear Regression":
        MODEL_PATH = BASE_DIR/config['paths']['model_2']

    elif model_type == "XGBoost":
        MODEL_PATH = BASE_DIR/config['paths']['model_3']
        
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Did you run training?")
    
    model = joblib.load(MODEL_PATH)
    return model
