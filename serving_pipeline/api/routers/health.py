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


__all__ = ["router"]

