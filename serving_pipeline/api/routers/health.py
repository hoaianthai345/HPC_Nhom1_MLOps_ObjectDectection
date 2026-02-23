"""
Routers for health and basic info endpoints.
"""
from fastapi import APIRouter

from ...config import settings
from .. import dependencies as deps
from ..schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Object Detection API", "docs": "/docs"}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_loaded=deps.is_model_loaded(),
    )


@router.get("/gpu/available")
async def check_gpu_availability():
    """Check if GPU detector is available."""
    return {
        "gpu_available": deps.is_gpu_available(),
        "message": "GPU detector is ready" if deps.is_gpu_available() 
                   else "GPU not available. CUDA may not be installed or GPU not present."
    }


@router.post("/tensorrt/reload")
async def reload_tensorrt_engine():
    """
    Reload TensorRT engine from MinIO.
    
    This endpoint is typically triggered by the convert_tensorrt DAG
    after successfully converting and uploading a new TensorRT engine.
    """
    result = deps.reload_tensorrt_detector()
    
    if result["status"] == "error":
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result


@router.get("/tensorrt/info")
async def get_tensorrt_info():
    """
    Get TensorRT engine information.
    
    Returns engine metadata including version, S3 key, size, and last modified timestamp.
    """
    return deps.get_tensorrt_info()


__all__ = ["router"]

