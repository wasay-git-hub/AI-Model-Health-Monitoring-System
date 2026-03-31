from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.fast_api.api_history import append_metrics_history, read_metrics_history
from src.fast_api.api_model_loader import load_model_once
from src.fast_api.api_metrics_service import compare_input_files, evaluate_input_file
from src.fast_api.api_schemas import (
    CompareInputFilesRequest,
    CompareInputFilesResponse,
    EvaluateFileRequest,
    EvaluateFileResponse,
    FileComparisonResult,
    FileMetrics,
    MetricsHistoryRecord,
    MetricRanking,
    PredictHealthRequest,
    PredictHealthResponse,
)
from src.utils import load_params

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize API runtime state once at startup."""
    config = load_params()
    loaded_model, model_type, model_path = load_model_once(config)

    app.state.config = config
    app.state.model = loaded_model
    app.state.model_loaded = True
    app.state.model_type = model_type
    app.state.model_artifact_path = str(model_path)
    app.state.metrics_history_path = BASE_DIR / "data" / "metrics_history.jsonl"

    # Metrics history file is the persistent store for past evaluations.
    app.state.metrics_history_path.parent.mkdir(parents=True, exist_ok=True)
    app.state.metrics_history_path.touch(exist_ok=True)

    yield

app = FastAPI(
    title="Rossmann Model Health API",
    description="API service for prediction and file-level metric evaluation/comparison.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", summary="Service health check")
def health_check():
    """Return API and model readiness state."""
    return {
        "status": "ok",
        "model_loaded": bool(app.state.model_loaded),
        "model_type": app.state.model_type,
    }

@app.get("/model-info", summary="Model metadata")
def model_info():
    """Return loaded model metadata and expected feature contract."""
    training_cfg = app.state.config["training_data"]
    return {
        "model_loaded": bool(app.state.model_loaded),
        "model_type": app.state.model_type,
        "model_artifact_path": app.state.model_artifact_path,
        "target": training_cfg["target"],
        "features": training_cfg["features"],
        "metrics_history_path": str(app.state.metrics_history_path),
    }

@app.post(
    "/predict-health",
    response_model=PredictHealthResponse,
    summary="Predict sales for one feature payload",
)
def predict_health(payload: PredictHealthRequest):
    """Predict sales value from one validated model-ready feature object."""
    feature_order = app.state.config["training_data"]["features"]
    row = payload.features.model_dump()
    feature_df = pd.DataFrame([row])[feature_order]

    prediction = float(app.state.model.predict(feature_df)[0])

    return PredictHealthResponse(
        prediction=prediction,
        model_type=app.state.model_type,
        timestamp=datetime.utcnow(),
    )

@app.post(
    "/evaluate-file",
    response_model=EvaluateFileResponse,
    summary="Evaluate one input CSV and compute all metrics",
)
def evaluate_file(payload: EvaluateFileRequest):
    """Run preprocessing + prediction for one CSV and return metric scores."""
    try:
        result = evaluate_input_file(payload.input_file, app.state.config, app.state.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = EvaluateFileResponse(
        input_file=result["input_file"],
        row_count=result["row_count"],
        metrics=FileMetrics(**result["metrics"]),
        model_type=app.state.model_type,
        timestamp=datetime.utcnow(),
    )

    append_metrics_history(
        history_path=app.state.metrics_history_path,
        endpoint="evaluate-file",
        model_type=app.state.model_type,
        input_files=[result["input_file"]],
        metrics_map={result["input_file"]: result["metrics"]},
    )

    return response

@app.post(
    "/compare-input-files",
    response_model=CompareInputFilesResponse,
    summary="Compare metrics across multiple input CSV files",
)
def compare_files(payload: CompareInputFilesRequest):
    """Evaluate multiple files and return per-metric best/worst ranking."""
    try:
        compared = compare_input_files(payload.input_files, app.state.config, app.state.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = CompareInputFilesResponse(
        model_type=app.state.model_type,
        total_files=len(compared["results"]),
        results=[
            FileComparisonResult(
                input_file=item["input_file"],
                row_count=item["row_count"],
                metrics=FileMetrics(**item["metrics"]),
            )
            for item in compared["results"]
        ],
        rankings=[MetricRanking(**ranking) for ranking in compared["rankings"]],
        timestamp=datetime.utcnow(),
    )

    metrics_map = {item["input_file"]: item["metrics"] for item in compared["results"]}
    input_files = [item["input_file"] for item in compared["results"]]

    append_metrics_history(
        history_path=app.state.metrics_history_path,
        endpoint="compare-input-files",
        model_type=app.state.model_type,
        input_files=input_files,
        metrics_map=metrics_map,
    )

    return response

@app.get(
    "/metrics-history",
    response_model=List[MetricsHistoryRecord],
    summary="Recent metric evaluation/comparison history",
)
def metrics_history(limit: int = Query(default=20, ge=1, le=200)):
    """Return recent metrics records from persistent JSONL history."""
    records = read_metrics_history(app.state.metrics_history_path, limit=limit)
    return [MetricsHistoryRecord(**record) for record in records]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "details": str(exc),
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": "HTTP_ERROR",
            "message": "Request could not be completed.",
            "details": str(exc.detail),
        },
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "Unexpected server error occurred.",
            "details": str(exc),
        },
    )