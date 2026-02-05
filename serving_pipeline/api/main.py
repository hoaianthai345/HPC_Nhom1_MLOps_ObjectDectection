"""
FastAPI application entrypoint for the Object Detection API.
"""
import asyncio
import logging

import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from ..config import settings
from . import dependencies as deps
from .routers.detection import router as detection_router
from .routers.health import router as health_router
from .routers.drift import router as drift_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Prometheus Gauges for system resources
PROCESS_RAM_MB = Gauge("service_ram_mb", "RAM used by FastAPI service in MB")
GPU_MEMORY_MB = Gauge(
    "gpu_memory_used_mb",
    "GPU memory used in MB",
    ["gpu_index"],
)

try:  # optional GPU metrics
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _HAS_GPU = True
except Exception:  # pragma: no cover - runs even without GPU
    _HAS_GPU = False


async def _metrics_collector() -> None:
    """Background task to collect RAM/GPU usage metrics."""
    process = psutil.Process()
    while True:
        try:
            mem_mb = process.memory_info().rss / (1024 * 1024)
            PROCESS_RAM_MB.set(mem_mb)

            if _HAS_GPU:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    for idx in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        GPU_MEMORY_MB.labels(gpu_index=idx).set(
                            info.used / (1024 * 1024)
                        )
                except Exception:
                    # don't crash metrics loop on GPU error
                    pass
        except Exception:
            # swallow errors, metrics are best-effort
            pass

        await asyncio.sleep(5)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Object Detection API using YOLO model",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app, include_in_schema=False)


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting Object Detection Service...")
    deps.init_components()
    # start background metrics collector
    asyncio.create_task(_metrics_collector())
    logger.info("Service startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Object Detection Service...")
    deps.shutdown_components()


# Register routers
app.include_router(health_router)
app.include_router(drift_router)
app.include_router(detection_router)


__all__ = ["app"]

