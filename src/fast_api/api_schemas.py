from datetime import datetime
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

class PredictionFeatures(BaseModel):
    """Model-ready feature payload for a single prediction request."""

    model_config = ConfigDict(extra="forbid")

    Store: int = Field(..., ge=1, description="Store identifier")
    DayOfWeek: int = Field(..., ge=1, le=7, description="Day index from 1 to 7")
    Promo: int = Field(..., ge=0, le=1, description="Promotion flag")
    StateHoliday: int = Field(..., ge=0, le=4, description="Mapped holiday category")
    SchoolHoliday: int = Field(..., ge=0, le=1, description="School holiday flag")
    StoreType: int = Field(..., ge=0, le=4, description="Mapped store type")
    Assortment: int = Field(..., ge=0, le=4, description="Mapped assortment type")
    CompetitionDistance: float = Field(..., ge=0, description="Distance to nearest competitor")
    Year: int = Field(..., ge=1900, le=2100, description="Calendar year")
    Month: int = Field(..., ge=1, le=12, description="Calendar month")
    Day: int = Field(..., ge=1, le=31, description="Day of month")

class PredictHealthRequest(BaseModel):
    """Request schema for POST /predict-health."""

    model_config = ConfigDict(extra="forbid")
    features: PredictionFeatures

class PredictHealthResponse(BaseModel):
    """Response schema for POST /predict-health."""

    prediction: float
    model_type: str
    timestamp: datetime
    prediction_id: int

class EvaluateFileRequest(BaseModel):
    """Request schema for POST /evaluate-file."""

    model_config = ConfigDict(extra="forbid")

    input_file: str = Field(..., min_length=1, description="Path to input CSV file")
    include_predictions: bool = Field(
        False,
        description="If true, response includes a lightweight prediction summary",
    )

class FileMetrics(BaseModel):
    """Evaluation metrics for one file."""

    RMSPE: float
    MAE: float
    RMSE: float
    MAPE: float
    R2: float

class EvaluateFileResponse(BaseModel):
    """Response schema for POST /evaluate-file."""

    input_file: str
    row_count: int
    metrics: FileMetrics
    model_type: str
    timestamp: datetime

class CompareInputFilesRequest(BaseModel):
    """Request schema for POST /compare-input-files."""

    model_config = ConfigDict(extra="forbid")

    input_files: List[str] = Field(..., min_length=2, description="CSV file paths to compare")

class FileComparisonResult(BaseModel):
    """Per-file metrics returned by comparison endpoint."""

    input_file: str
    row_count: int
    metrics: FileMetrics

class MetricRanking(BaseModel):
    """Best and worst file for a given metric."""

    metric: Literal["RMSPE", "MAE", "RMSE", "MAPE", "R2"]
    best_file: str
    worst_file: str

class CompareInputFilesResponse(BaseModel):
    """Response schema for POST /compare-input-files."""

    model_type: str
    total_files: int
    results: List[FileComparisonResult]
    rankings: List[MetricRanking]
    timestamp: datetime

class MetricsHistoryRecord(BaseModel):
    """Persistent record format used by metrics history endpoint and storage."""

    run_id: str
    timestamp: datetime
    endpoint: Literal["evaluate-file", "compare-input-files"]
    model_type: str
    input_files: List[str]
    metrics: Dict[str, Dict[str, float]]
    notes: Optional[str] = None

class ErrorResponse(BaseModel):
    """Consistent API error payload."""

    error_code: str
    message: str
    details: Optional[str] = None