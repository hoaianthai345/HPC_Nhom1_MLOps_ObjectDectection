"""
Pydantic schemas used by the FastAPI service.
"""
from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    center_x: float
    center_y: float


class DetectionItem(BaseModel):
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


class ImageSize(BaseModel):
    width: int
    height: int


class DetectionResponse(BaseModel):
    request_id: str
    num_detections: int
    image_size: ImageSize
    inference_time_ms: float
    detections: List[DetectionItem]


class ErrorResponse(BaseModel):
    error: str
    detail: str


class DriftRequest(BaseModel):
    """Request body for data drift analysis."""

    data_dir: str = "data_final"
    train_split: str = "train"
    test_split: str = "valid"
    output_dir: str = "reports"
    batch_size: int = 32
    img_size: int = 640
    class_names: Optional[List[str]] = None
    max_samples: Optional[int] = None


class DriftResponse(BaseModel):
    """Response for data drift analysis."""

    passed: bool
    report_path: str


class ModelPerformanceRequest(BaseModel):
    """Request body for model performance analysis."""

    data_dir: str = "data_final"
    split: str = "valid"
    output_dir: str = "reports"
    batch_size: int = 32
    img_size: int = 640
    class_names: Optional[List[str]] = None
    max_samples: Optional[int] = None


class ModelPerformanceResponse(BaseModel):
    """Response for model performance analysis."""

    passed: bool
    report_score: Optional[float] = None
    report_path: str


__all__ = [
    "HealthResponse",
    "BoundingBox",
    "DetectionItem",
    "ImageSize",
    "DetectionResponse",
    "ErrorResponse",
    "DriftRequest",
    "DriftResponse",
    "ModelPerformanceRequest",
    "ModelPerformanceResponse",
]

