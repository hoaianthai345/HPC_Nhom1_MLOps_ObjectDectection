"""
Configuration for Data Pipeline

Centralized configuration for MinIO, data paths, and pipeline settings.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Configuration management for data pipeline"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        load_dotenv()
        
        # MinIO Configuration
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
        self.minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minio_admin")
        self.minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minio_password123")
        self.minio_region = "us-east-1"
        
        # Bucket Names
        self.training_bucket = "training-data"
        self.production_bucket = "production-data"
        self.model_exports_bucket = "model-exports"
        
        # Project Paths
        self.project_root = Path(__file__).parent.parent
        self.data_root = self.project_root / "data"
        self.download_dir = self.data_root / "download"
        self.processed_dir = self.data_root / "processed"
        self.test_variants_dir = self.data_root / "test_variants"
        
        # Kaggle Configuration
        self.kaggle_token = os.getenv("KAGGLE_API_TOKEN")
        
        # Default Dataset Configuration
        self.default_kaggle_dataset = "yusufberksardoan/traffic-detection-project"
        self.default_classes = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in [self.data_root, self.download_dir, 
                         self.processed_dir, self.test_variants_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_minio_config(self) -> bool:
        """Validate MinIO configuration"""
        if not all([self.minio_endpoint, self.minio_access_key, self.minio_secret_key]):
            return False
        return True
    
    def validate_kaggle_config(self) -> bool:
        """Validate Kaggle configuration"""
        return self.kaggle_token is not None
    
    def get_boto3_config(self) -> dict:
        """Get boto3 client configuration for MinIO"""
        from botocore.client import Config as BotoConfig
        
        return {
            'endpoint_url': self.minio_endpoint,
            'aws_access_key_id': self.minio_access_key,
            'aws_secret_access_key': self.minio_secret_key,
            'config': BotoConfig(signature_version='s3v4'),
            'region_name': self.minio_region
        }
    
    def __repr__(self):
        return f"Config(minio={self.minio_endpoint}, training_bucket={self.training_bucket})"


# Global config instance
config = Config()
