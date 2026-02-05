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
from fastapi.responses import StreamingResponse
from PIL import Image

from ...config import settings
from ...models.yolo_model import YOLODetector, DetectionResult
from ...utils.validators import ImageValidator
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
    Detect objects in an uploaded image.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing detection request for: {file.filename}")

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
    "/detect/annotated",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect_and_draw(
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
    Detect objects and return annotated image with bounding boxes drawn.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing annotated detection request: {file.filename}")

    try:
        # Validate image
        image_bytes, width, height, img_format = await val.validate_upload_file(file)

        # Run detection and get annotated image
        result, annotated_image = det.predict_and_draw(
            image_bytes,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        logger.info(f"[{request_id}] Detection complete: {result.num_detections} objects")

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "X-Request-ID": request_id,
                "X-Num-Detections": str(result.num_detections),
                "X-Inference-Time-Ms": str(round(result.inference_time * 1000, 2)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Annotated detection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}",
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

