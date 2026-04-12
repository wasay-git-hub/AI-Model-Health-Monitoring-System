from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.utils import load_params
from src.fast_api.database import Session, ModelHealthLog
from src.fast_api.database import engine, Base, sessionmaker
from src.fast_api.api_model_loader import load_model_once
from src.fast_api.api_metrics_service import compare_input_files, evaluate_input_file
from src.fast_api.api_schemas import (
    CompareInputFilesRequest,
    CompareInputFilesResponse,
    EvaluateFileRequest,
    EvaluateFileResponse,
    FileComparisonResult,
    FileMetrics,
    MetricRanking,
    PredictHealthRequest,
    PredictHealthResponse,
)

import mlflow
import time
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize API runtime state once at startup."""
    config = load_params()
    loaded_model, model_type, model_path = load_model_once(config)

    # Initialize MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Sales_Monitoring")

    app.state.config = config
    app.state.model = loaded_model
    app.state.model_loaded = True
    app.state.model_type = model_type
    app.state.model_artifact_path = str(model_path)

    yield

app = FastAPI(
    title="Sales Model Health API",
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
    }

@app.post(
    "/predict-health",
    response_model=PredictHealthResponse,
    summary="Predict sales for one feature payload",
)
def predict_health(payload: PredictHealthRequest):
    """Predict sales value from one validated model-ready feature object and log latency."""
    start_time = time.perf_counter()
    
    try:
        feature_order = app.state.config["training_data"]["features"]
        row = payload.features.model_dump()
        feature_df = pd.DataFrame([row])[feature_order]

        prediction = float(app.state.model.predict(feature_df)[0])
        
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Database Logging
        db = Session()
        try:
            new_log = ModelHealthLog(
                endpoint="predict-health",
                model_type=app.state.model_type,
                dataset_source="single_request",
                row_count=1,
                latency_ms=round(duration_ms, 2)
            )
            db.add(new_log)
            db.commit()
        except Exception as e:
            print(f"Database logging failed: {e}")
        finally:
            db.close()

        return PredictHealthResponse(
            prediction=prediction,
            model_type=app.state.model_type,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/evaluate-file",
    response_model=EvaluateFileResponse,
    summary="Evaluate one input CSV and compute all metrics",
)
def evaluate_file(payload: EvaluateFileRequest):
    """Run preprocessing + prediction for one CSV, compute metrics, and log to MLFlow & Supabase."""
    start_time = time.perf_counter()
    
    try:
        # Core processing logic
        result = evaluate_input_file(payload.input_file, app.state.config, app.state.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # MLFlow Logging
    with mlflow.start_run(run_name=f"Eval_{Path(payload.input_file).name}"):
        mlflow.log_param("model_type", app.state.model_type)
        mlflow.log_param("input_file", payload.input_file)
        for metric_name, value in result["metrics"].items():
            mlflow.log_metric(metric_name, value)
        mlflow.log_metric("latency_ms", duration_ms)

    # Database Logging (Supabase)
    db = Session()
    try:
        new_log = ModelHealthLog(
            endpoint="evaluate-file",
            model_type=app.state.model_type,
            dataset_source=result["input_file"],
            row_count=result["row_count"],
            rmspe=result["metrics"]["RMSPE"],
            mae=result["metrics"]["MAE"],
            mape=result["metrics"]["MAPE"],
            rmse=result["metrics"]["RMSE"],
            r2_score=result["metrics"]["R2"],
            latency_ms=round(duration_ms, 2)
        )
        db.add(new_log)
        db.commit()
    except Exception as e:
        print(f"Database logging failed: {e}")
    finally:
        db.close()

    return EvaluateFileResponse(
        input_file=result["input_file"],
        row_count=result["row_count"],
        metrics=FileMetrics(**result["metrics"]),
        model_type=app.state.model_type,
        timestamp=datetime.now(timezone.utc)
    )

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
        timestamp=datetime.now(timezone.utc),
    )

    # Database Logging
    db = Session()
    try:
        # Loop through each file result and save to Supabase
        for item in compared["results"]:
            new_log = ModelHealthLog(
                endpoint="compare-input-files",
                model_type=app.state.model_type,
                dataset_source=item["input_file"],
                row_count=item["row_count"],
                rmspe=item["metrics"]["RMSPE"],
                mae=item["metrics"]["MAE"],
                mape=item["metrics"]["MAPE"],
                rmse=item["metrics"]["RMSE"],
                r2_score=item["metrics"]["R2"]
            )
            db.add(new_log)
        
        db.commit()
        print(f"Comparison metrics logged to Supabase for {len(compared['results'])} files.")
    except Exception as e:
        print(f"Database logging failed during comparison: {e}")
    finally:
        db.close()

    return response

@app.get("/metrics-history", summary="Recent metric evaluation/comparison history")
def metrics_history(limit: int = Query(default=20, ge=1, le=200)):
    """Return recent metrics records from Supabase cloud database."""
    db = Session()
    try:
        # Query the database, sort by newest first, and get the list
        logs = db.query(ModelHealthLog).order_by(ModelHealthLog.timestamp.desc()).limit(limit).all()
        return logs
    finally:
        db.close()

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
        }
    )