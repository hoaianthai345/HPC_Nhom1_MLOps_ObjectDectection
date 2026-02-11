from typing import Optional

from fastapi import HTTPException

from ..models.yolo_model import YOLODetector
from ..utils.validators import ImageValidator
from ..config import settings

_detector: Optional[YOLODetector] = None
_gpu_detector: Optional[YOLODetector] = None
_validator: Optional[ImageValidator] = None


def init_components() -> None:
    """Initialize singleton detector and validator."""
    global _detector, _gpu_detector, _validator
    _validator = ImageValidator()
    
    # Initialize CPU detector
    _detector = YOLODetector(device="cpu")
    
    # Initialize GPU detector if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 CUDA is available, initializing GPU detector...")
            _gpu_detector = YOLODetector(device="cuda")
            print(f"✅ GPU detector initialized on device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, GPU detector will not be initialized")
    except ImportError:
        print("⚠️  PyTorch not available, GPU detector will not be initialized")
    except Exception as e:
        print(f"⚠️  Error initializing GPU detector: {e}")


def shutdown_components() -> None:
    """Cleanup resources on shutdown (placeholder for future)."""
    # Currently nothing to explicitly cleanup.
    pass


def get_detector() -> YOLODetector:
    """FastAPI dependency to get initialized CPU detector."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return _detector


def get_gpu_detector() -> YOLODetector:
    """FastAPI dependency to get initialized GPU detector."""
    if _gpu_detector is None:
        raise HTTPException(
            status_code=503,
            detail="GPU detector not available. CUDA may not be installed or GPU not present."
        )
    return _gpu_detector


def get_validator() -> ImageValidator:
    """FastAPI dependency to get initialized image validator."""
    if _validator is None:
        raise HTTPException(status_code=503, detail="Validator not initialized")
    return _validator


def is_model_loaded() -> bool:
    return _detector is not None


def is_gpu_available() -> bool:
    """Check if GPU detector is available."""
    return _gpu_detector is not None


__all__ = [
    "init_components",
    "shutdown_components",
    "get_detector",
    "get_gpu_detector",
    "get_validator",
    "is_model_loaded",
    "is_gpu_available",
]

