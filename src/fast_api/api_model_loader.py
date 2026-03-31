from pathlib import Path
from src.model_pipeline.model_serializer import load_model

def resolve_model_artifact_path(config, model_type):
    """Return the configured model artifact path for the selected model type."""
    base_dir = Path(__file__).resolve().parent.parent.parent

    if model_type == "Random Forest":
        return base_dir / config["paths"]["model_1"]

    if model_type == "Linear Regression":
        return base_dir / config["paths"]["model_2"]

    if model_type == "XGBoost":
        return base_dir / config["paths"]["model_3"]

    raise ValueError(f"Model type '{model_type}' is not supported.")

def load_model_once(config):
    """Load the configured model artifact for API startup initialization."""
    model_type = config["models"]["type"]
    artifact_path = resolve_model_artifact_path(config, model_type)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. Run training first."
        )

    model = load_model(model_type)
    return model, model_type, artifact_path