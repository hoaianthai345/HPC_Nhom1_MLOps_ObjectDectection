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


__all__ = ["router"]

