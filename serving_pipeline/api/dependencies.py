"""
Shared FastAPI dependencies for the API layer.
"""
from typing import Optional

from fastapi import HTTPException

from ..models.yolo_model import YOLODetector
from ..utils.validators import ImageValidator

_detector: Optional[YOLODetector] = None
_validator: Optional[ImageValidator] = None


def init_components() -> None:
    """Initialize singleton detector and validator."""
    global _detector, _validator
    _validator = ImageValidator()
    _detector = YOLODetector()


def shutdown_components() -> None:
    """Cleanup resources on shutdown (placeholder for future)."""
    # Currently nothing to explicitly cleanup.
    pass


def get_detector() -> YOLODetector:
    """FastAPI dependency to get initialized detector."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return _detector


def get_validator() -> ImageValidator:
    """FastAPI dependency to get initialized image validator."""
    if _validator is None:
        raise HTTPException(status_code=503, detail="Validator not initialized")
    return _validator


def is_model_loaded() -> bool:
    return _detector is not None


__all__ = [
    "init_components",
    "shutdown_components",
    "get_detector",
    "get_validator",
    "is_model_loaded",
]

