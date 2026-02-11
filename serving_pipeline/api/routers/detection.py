"""
Routers for detection-related endpoints.
"""
import io
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from PIL import Image

from ...config import settings
from ...models.yolo_model import YOLODetector, DetectionResult
from ...utils.validators import ImageValidator
from ...utils.minio_client import get_minio_client
from .. import dependencies as deps
from ..schemas import (
    DetectionResponse,
    ErrorResponse,
    ImageSize,
    DetectionItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Detection"])


def _save_production_sample(
    image_bytes: bytes,
    result: DetectionResult,
    original_filename: str,
) -> None:
    """
    Save uploaded image + YOLO-format predictions into production folder.

    Structure (under settings.PRODUCTION_DIR):
      - images/<stem>.jpg
      - predictions/<stem>.txt
    """
    try:
        prod_root = settings.PRODUCTION_DIR
        images_dir = prod_root / "images"
        preds_dir = prod_root / "predictions"
        images_dir.mkdir(parents=True, exist_ok=True)
        preds_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(original_filename).stem or "image"
        base_name = f"{stem}_{timestamp}"

        # Save image as JPEG
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = images_dir / f"{base_name}.jpg"
        img.save(image_path, format="JPEG", quality=95)

        # Save predictions in YOLO txt format: class x_center y_center w h conf
        pred_path = preds_dir / f"{base_name}.txt"
        with pred_path.open("w") as f:
            for det in result.detections:
                x_center = det.center_x / result.image_width
                y_center = det.center_y / result.image_height
                width = det.width / result.image_width
                height = det.height / result.image_height
                line = (
                    f"{det.class_id} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f} "
                    f"{det.confidence:.4f}\n"
                )
                f.write(line)

        logger.info(
            "Saved production sample: image=%s, preds=%s",
            image_path,
            pred_path,
        )
    except Exception as exc:
        # Do not break API if saving fails
        logger.warning(f"Failed to save production sample: {exc}")


def _save_request_image_to_bucket(
    image_bytes: bytes,
    original_filename: str,
    result: Optional[DetectionResult] = None,
) -> None:
    """
    Save user request image and student predictions to production-data bucket organized by date.
    
    Supports two modes:
    1. MinIO (if USE_MINIO=True): Saves to MinIO bucket
    2. Local (if USE_MINIO=False): Saves to local filesystem
    
    Structure:
      - YYYY-MM-DD/<unique_id>_<original_name>.jpg (image)
      - YYYY-MM-DD/<unique_id>_<original_name>.txt (student predictions in YOLO format)
    
    This allows cron jobs/DAGs to process images and compare student vs teacher predictions.
    """
    try:
        # Get current date for folder organization
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create unique filename with timestamp and UUID
        timestamp = datetime.now().strftime("%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        stem = Path(original_filename).stem or "image"
        filename = f"{unique_id}_{stem}_{timestamp}.jpg"
        object_key = f"{current_date}/{filename}"
        
        # Convert image bytes to PIL Image and save as JPEG bytes
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        img_bytes = img_buffer.getvalue()
        
        # Save to MinIO or local storage
        if settings.USE_MINIO:
            minio_client = get_minio_client()
            if minio_client:
                # Save image
                success = minio_client.upload_bytes(
                    bucket_name=settings.MINIO_PRODUCTION_BUCKET,
                    object_key=object_key,
                    data=img_bytes
                )
                if success:
                    logger.info(
                        "Saved request image to MinIO: %s/%s (date: %s)",
                        settings.MINIO_PRODUCTION_BUCKET,
                        object_key,
                        current_date,
                    )
                else:
                    logger.warning("Failed to save image to MinIO")
                
                # Save student predictions if provided
                if result and success:
                    pred_filename = Path(filename).stem + ".txt"
                    pred_object_key = f"{current_date}/{pred_filename}"
                    
                    # Create YOLO format predictions
                    pred_lines = []
                    for det in result.detections:
                        x_center = det.center_x / result.image_width
                        y_center = det.center_y / result.image_height
                        width = det.width / result.image_width
                        height = det.height / result.image_height
                        line = (
                            f"{det.class_id} "
                            f"{x_center:.6f} {y_center:.6f} "
                            f"{width:.6f} {height:.6f} "
                            f"{det.confidence:.4f}\n"
                        )
                        pred_lines.append(line)
                    
                    pred_bytes = "".join(pred_lines).encode('utf-8')
                    pred_success = minio_client.upload_bytes(
                        bucket_name=settings.MINIO_PRODUCTION_BUCKET,
                        object_key=pred_object_key,
                        data=pred_bytes
                    )
                    if pred_success:
                        logger.info(
                            "Saved student predictions to MinIO: %s/%s",
                            settings.MINIO_PRODUCTION_BUCKET,
                            pred_object_key,
                        )
            else:
                logger.warning("MinIO client not available, skipping bucket save")
        else:
            # Save to local filesystem
            try:
                date_folder = settings.PRODUCTION_DATA_DIR / current_date
                date_folder.mkdir(parents=True, exist_ok=True)
                
                # Save image
                image_path = date_folder / filename
                img.save(image_path, format="JPEG", quality=95)
                logger.info(
                    "Saved request image to local storage: %s (date: %s)",
                    image_path,
                    current_date,
                )
                
                # Save student predictions if provided
                if result:
                    pred_filename = Path(filename).stem + ".txt"
                    pred_path = date_folder / pred_filename
                    with pred_path.open("w") as f:
                        for det in result.detections:
                            x_center = det.center_x / result.image_width
                            y_center = det.center_y / result.image_height
                            width = det.width / result.image_width
                            height = det.height / result.image_height
                            line = (
                                f"{det.class_id} "
                                f"{x_center:.6f} {y_center:.6f} "
                                f"{width:.6f} {height:.6f} "
                                f"{det.confidence:.4f}\n"
                            )
                            f.write(line)
                    logger.info(
                        "Saved student predictions to local storage: %s",
                        pred_path,
                    )
            except (OSError, PermissionError) as e:
                logger.warning(
                    f"Could not save to local filesystem: {e}. "
                    "Consider using MinIO storage (USE_MINIO=true)"
                )
        
    except Exception as exc:
        # Do not break API if saving fails
        logger.warning(f"Failed to save request image to bucket: {exc}")


@router.post(
    "/detect",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect_objects(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Confidence threshold for detections"
    ),
    iou_threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="IoU threshold for NMS"
    ),
    det: YOLODetector = Depends(deps.get_detector),
    val: ImageValidator = Depends(deps.get_validator),
):
    """
    Detect objects in an uploaded image using CPU.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing CPU detection request for: {file.filename}")

    try:
        # Validate image
        image_bytes, width, height, img_format = await val.validate_upload_file(file)
        logger.info(f"[{request_id}] Image validated: {width}x{height}, format: {img_format}")

        # Run detection
        result = det.predict(
            image_bytes,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        logger.info(f"[{request_id}] Detection complete: {result.num_detections} objects found")

        # Save sample + predictions in production folder (YOLO format)
        _save_production_sample(
            image_bytes=image_bytes,
            result=result,
            original_filename=file.filename or f"{request_id}.jpg",
        )
        
        # Save user request image and student predictions to production-data bucket organized by date
        _save_request_image_to_bucket(
            image_bytes=image_bytes,
            original_filename=file.filename or f"{request_id}.jpg",
            result=result,
        )

        # Build response
        result_dict = result.to_dict()

        return DetectionResponse(
            request_id=request_id,
            num_detections=result_dict["num_detections"],
            image_size=ImageSize(**result_dict["image_size"]),
            inference_time_ms=result_dict["inference_time_ms"],
            detections=[DetectionItem(**d) for d in result_dict["detections"]],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Detection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}",
        )


@router.post(
    "/detect-gpu",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def detect_objects_gpu(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Confidence threshold for detections"
    ),
    iou_threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="IoU threshold for NMS"
    ),
    det: YOLODetector = Depends(deps.get_gpu_detector),
    val: ImageValidator = Depends(deps.get_validator),
):
    """
    Detect objects in an uploaded image using GPU acceleration.
    
    This endpoint requires CUDA-enabled GPU and will return 503 if GPU is not available.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing GPU detection request for: {file.filename}")

    try:
        # Validate image
        image_bytes, width, height, img_format = await val.validate_upload_file(file)
        logger.info(f"[{request_id}] Image validated: {width}x{height}, format: {img_format}")

        # Run detection on GPU
        result = det.predict(
            image_bytes,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        logger.info(f"[{request_id}] GPU detection complete: {result.num_detections} objects found")

        # Save sample + predictions in production folder (YOLO format)
        _save_production_sample(
            image_bytes=image_bytes,
            result=result,
            original_filename=file.filename or f"{request_id}.jpg",
        )
        
        # Save user request image and student predictions to production-data bucket organized by date
        _save_request_image_to_bucket(
            image_bytes=image_bytes,
            original_filename=file.filename or f"{request_id}.jpg",
            result=result,
        )

        # Build response
        result_dict = result.to_dict()

        return DetectionResponse(
            request_id=request_id,
            num_detections=result_dict["num_detections"],
            image_size=ImageSize(**result_dict["image_size"]),
            inference_time_ms=result_dict["inference_time_ms"],
            detections=[DetectionItem(**d) for d in result_dict["detections"]],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] GPU detection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"GPU detection failed: {str(e)}",
        )


@router.get("/model/info", tags=["Model"])
async def get_model_info(det: YOLODetector = Depends(deps.get_detector)):
    """Get information about the loaded model."""
    return {
        "model_path": settings.YOLO_MODEL_PATH,
        "confidence_threshold": settings.YOLO_CONFIDENCE_THRESHOLD,
        "iou_threshold": settings.YOLO_IOU_THRESHOLD,
        "max_detections": settings.YOLO_MAX_DETECTIONS,
        "class_names": det.class_names,
    }


@router.get("/config", tags=["Config"])
async def get_config():
    """Get current configuration (non-sensitive)."""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "image_constraints": {
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "min_dimensions": f"{settings.MIN_IMAGE_WIDTH}x{settings.MIN_IMAGE_HEIGHT}",
            "max_dimensions": f"{settings.MAX_IMAGE_WIDTH}x{settings.MAX_IMAGE_HEIGHT}",
        },
    }


__all__ = ["router"]

