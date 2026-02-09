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
    API_PORT: int = Field(default=8000, env="API_PORT")
    GRADIO_PORT: int = Field(default=7860, env="GRADIO_PORT")
    
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
    
    # Folder to store production samples (images + YOLO txt predictions)
    PRODUCTION_DIR: Path = Field(
        default=Path(__file__).parent / "production",
        env="PRODUCTION_DIR",
    )
    
    model_config = {
        "env_file": str(Path(__file__).parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        (self.PRODUCTION_DIR / "images").mkdir(parents=True, exist_ok=True)
        (self.PRODUCTION_DIR / "predictions").mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.setup_directories()
