from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import pandas as pd
import mlflow
import time
import os

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.utils import load_params
from src.fast_api.database import Session, ModelHealthLog, SingleInferenceLog
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_params()
    loaded_model, model_type, model_path = load_model_once(config)

    # This tells MLflow: "If there is no network server, just save to this local folder"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    app.state.config = config
    app.state.model = loaded_model
    app.state.model_loaded = True
    app.state.model_type = model_type
    app.state.model_artifact_path = str(model_path)
    yield

app = FastAPI(
    title="Sales Model Health API",
    description="Full-stack AI service with Cloud Monitoring and Feedback Loops.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UI ROUTES ---
app.mount("/static", StaticFiles(directory="src/frontend"), name="static")

@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False) # Add this
async def read_index():
    return FileResponse("src/frontend/index.html")

@app.get("/predict.html", include_in_schema=False)
async def read_predict():
    return FileResponse("src/frontend/predict.html")

@app.get("/evaluate.html", include_in_schema=False)
async def read_evaluate():
    return FileResponse("src/frontend/evaluate.html")

# --- LOGIC ENDPOINTS ---

@app.get("/health", summary="Service health check")
def health_check():
    return {
        "status": "ok",
        "model_loaded": bool(app.state.model_loaded),
        "model_type": app.state.model_type,
    }

@app.get("/model-info", summary="Model metadata")
def model_info():
    training_cfg = app.state.config["training_data"]
    return {
        "model_loaded": bool(app.state.model_loaded),
        "model_type": app.state.model_type,
        "target": training_cfg["target"],
        "features": training_cfg["features"],
    }

@app.post("/predict-health", response_model=PredictHealthResponse, summary="Single prediction")
def predict_health(payload: PredictHealthRequest):
    start_time = time.perf_counter()
    try:
        feature_order = app.state.config["training_data"]["features"]
        feature_dict = payload.features.model_dump()
        feature_df = pd.DataFrame([feature_dict])[feature_order]
        prediction = float(app.state.model.predict(feature_df)[0])
        
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        db = Session()
        try:
            new_inference = SingleInferenceLog(
                model_type=app.state.model_type,
                inputs=feature_dict,
                prediction_value=prediction,
                latency_ms=round(duration_ms, 2)
            )
            db.add(new_inference)
            db.commit()
            db.refresh(new_inference)
            prediction_id = new_inference.inference_id
        finally:
            db.close()

        return PredictHealthResponse(
            prediction=prediction,
            prediction_id=prediction_id,
            model_type=app.state.model_type,
            timestamp=datetime.now(timezone.utc)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/evaluate-file", response_model=EvaluateFileResponse, summary="Batch evaluation")
def evaluate_file(payload: EvaluateFileRequest):
    start_time = time.perf_counter()
    try:
        result = evaluate_input_file(payload.input_file, app.state.config, app.state.model)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    with mlflow.start_run(run_name=f"Eval_{Path(payload.input_file).name}"):
        for metric_name, value in result["metrics"].items():
            mlflow.log_metric(metric_name, value)
        mlflow.log_metric("latency_ms", duration_ms)

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
    finally:
        db.close()

    return EvaluateFileResponse(
        input_file=result["input_file"],
        row_count=result["row_count"],
        metrics=FileMetrics(**result["metrics"]),
        model_type=app.state.model_type,
        timestamp=datetime.now(timezone.utc)
    )

@app.post("/feedback")
def submit_feedback(prediction_id: int, actual_sales: float):
    # This matches the JS call: /feedback?prediction_id=...&actual_sales=...
    db = Session()
    try:
        record = db.query(SingleInferenceLog).filter(SingleInferenceLog.inference_id == prediction_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Prediction ID not found")
        
        record.actual_sales = actual_sales
        db.commit()
        return {"status": "success"}
    finally:
        db.close()

@app.get("/metrics-history")
def metrics_history(limit: int = Query(default=20, ge=1, le=200)):
    db = Session()
    try:
        # We add a print here to see if the function is even starting
        print("Fetching history from Supabase...") 
        data = db.query(ModelHealthLog).order_by(ModelHealthLog.timestamp.desc()).limit(limit).all()
        print(f"Successfully fetched {len(data)} rows.")
        return data
    except Exception as e:
        print(f"CRITICAL SQL ERROR: {e}") # This will show the real error in your terminal
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# --- EXCEPTION HANDLERS ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error_code": "VALIDATION_ERROR", "details": str(exc)})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error_code": "INTERNAL_SERVER_ERROR", "details": str(exc)})