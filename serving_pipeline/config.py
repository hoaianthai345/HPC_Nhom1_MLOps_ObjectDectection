from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # App settings
    APP_NAME: str = "Object Detection Service"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    GRADIO_PORT: int = Field(default=7860, env="GRADIO_PORT")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_HOST: str = Field(default="localhost", env="API_HOST")  # For Gradio to connect to API
    
    # GPU Service settings (separate service)
    GPU_API_HOST: str = Field(default="localhost", env="GPU_API_HOST")  # For Gradio to connect to GPU API
    GPU_API_PORT: int = Field(default=8001, env="GPU_API_PORT")  # GPU service port
    
    # Image validation settings
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    MAX_FILE_SIZE_MB: float = Field(default=10.0, env="MAX_FILE_SIZE_MB")
    MIN_IMAGE_WIDTH: int = Field(default=32, env="MIN_IMAGE_WIDTH")
    MIN_IMAGE_HEIGHT: int = Field(default=32, env="MIN_IMAGE_HEIGHT")
    MAX_IMAGE_WIDTH: int = Field(default=4096, env="MAX_IMAGE_WIDTH")
    MAX_IMAGE_HEIGHT: int = Field(default=4096, env="MAX_IMAGE_HEIGHT")
    
    # YOLO Model settings
    YOLO_MODEL_PATH: str = Field(default="yolov8n.pt", env="YOLO_MODEL_PATH")
    YOLO_CONFIDENCE_THRESHOLD: float = Field(default=0.25, env="YOLO_CONFIDENCE_THRESHOLD")
    YOLO_IOU_THRESHOLD: float = Field(default=0.45, env="YOLO_IOU_THRESHOLD")
    YOLO_MAX_DETECTIONS: int = Field(default=100, env="YOLO_MAX_DETECTIONS")
    DEVICE: str = Field(default="cpu", env="DEVICE")  # Device for inference: 'cpu' or 'cuda'
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    MLFLOW_MODEL_URI: str = Field(default="", env="MLFLOW_MODEL_URI")  # e.g., models:/yolo-student/Production
    MLFLOW_S3_ENDPOINT_URL: str = Field(default="http://localhost:9000", env="MLFLOW_S3_ENDPOINT_URL")
    AWS_ACCESS_KEY_ID: str = Field(default="minio_admin", env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(default="minio_password123", env="AWS_SECRET_ACCESS_KEY")
    
    # Folder to store production samples (images + YOLO txt predictions)
    PRODUCTION_DIR: Path = Field(
        default=Path("/data/production"),
        env="PRODUCTION_DIR",
    )
    
    # Folder to store user request images organized by date for teacher model inference
    PRODUCTION_DATA_DIR: Path = Field(
        default=Path("/data/production-data"),
        env="PRODUCTION_DATA_DIR",
    )
    
    # MinIO settings for production data storage
    MINIO_ENDPOINT: str = Field(default="http://localhost:9000", env="MINIO_ENDPOINT")
    MINIO_PRODUCTION_BUCKET: str = Field(default="production-data", env="MINIO_PRODUCTION_BUCKET")
    USE_MINIO: bool = Field(default=False, env="USE_MINIO")  # Use MinIO or local storage
    
    model_config = {
        "env_file": str(Path(__file__).parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Try to create directories, but don't fail if read-only filesystem
        directories = [
            (self.PRODUCTION_DIR, "production directory"),
            (self.PRODUCTION_DIR / "images", "production images directory"),
            (self.PRODUCTION_DIR / "predictions", "production predictions directory"),
        ]
        
        # Only create PRODUCTION_DATA_DIR if not using MinIO
        if not self.USE_MINIO:
            directories.append((self.PRODUCTION_DATA_DIR, "production data directory"))
        
        for directory, desc in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"✓ Created {desc}: {directory}")
            except (OSError, PermissionError) as e:
                logger.warning(
                    f"Could not create {desc} at {directory}: {e}. "
                    "This is normal if using read-only filesystem or MinIO storage."
                )


# Global settings instance
settings = Settings()
settings.setup_directories()
