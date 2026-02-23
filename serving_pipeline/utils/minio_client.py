"""
MinIO client utilities for production data storage.
"""
import io
import logging
import warnings
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from ..config import settings

# Suppress urllib3 header parsing warnings for MinIO compatibility
warnings.filterwarnings('ignore', message='Failed to parse headers')
logging.getLogger('urllib3.connection').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class MinIOClient:
    """Wrapper class for MinIO operations."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO endpoint URL (defaults to settings.MINIO_ENDPOINT)
            access_key: MinIO access key (defaults to settings.AWS_ACCESS_KEY_ID)
            secret_key: MinIO secret key (defaults to settings.AWS_SECRET_ACCESS_KEY)
        """
        self.endpoint = endpoint or settings.MINIO_ENDPOINT
        self.access_key = access_key or settings.AWS_ACCESS_KEY_ID
        self.secret_key = secret_key or settings.AWS_SECRET_ACCESS_KEY
        
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(
                    signature_version='s3v4',
                    connect_timeout=10,  # Connection timeout in seconds
                    read_timeout=60,     # Read timeout in seconds
                ),
                region_name='us-east-1'
            )
            logger.info(f"MinIO client initialized (endpoint: {self.endpoint})")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            raise
    
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        try:
            self.client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError:
            return False
        except Exception as e:
            logger.error(f"Error checking bucket {bucket_name}: {e}")
            return False
    
    def create_bucket(self, bucket_name: str) -> bool:
        """Create a new bucket if it doesn't exist."""
        try:
            if self.bucket_exists(bucket_name):
                logger.info(f"Bucket {bucket_name} already exists")
                return True
            
            self.client.create_bucket(Bucket=bucket_name)
            logger.info(f"✅ Created bucket: {bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating bucket {bucket_name}: {e}")
            return False
    
    def upload_file(self, bucket_name: str, object_key: str, file_path: str) -> bool:
        """
        Upload a file to MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_key: Object key (path) in the bucket
            file_path: Local file path to upload
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.upload_file(file_path, bucket_name, object_key)
            logger.debug(f"Uploaded {file_path} to {bucket_name}/{object_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading {file_path} to {bucket_name}/{object_key}: {e}")
            return False
    
    def upload_bytes(self, bucket_name: str, object_key: str, data: bytes) -> bool:
        """
        Upload bytes data to MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_key: Object key (path) in the bucket
            data: Bytes data to upload
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=data
            )
            logger.debug(f"Uploaded {len(data)} bytes to {bucket_name}/{object_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading bytes to {bucket_name}/{object_key}: {e}")
            return False
    
    def download_file(self, bucket_name: str, object_key: str, local_path: str) -> bool:
        """
        Download a file from MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_key: Object key (path) in the bucket
            local_path: Local file path to save to
        
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket_name, object_key, local_path)
            logger.debug(f"Downloaded {bucket_name}/{object_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {bucket_name}/{object_key}: {e}")
            return False
    
    def list_objects(self, bucket_name: str, prefix: str = '') -> list:
        """
        List all objects in a bucket with optional prefix.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Filter objects by prefix (folder path)
        
        Returns:
            List of object keys
        """
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    objects.extend([obj['Key'] for obj in page['Contents']])
            
            return objects
        except Exception as e:
            logger.error(f"Error listing objects in {bucket_name}/{prefix}: {e}")
            return []


# Global MinIO client instance (optional, lazy initialization)
_minio_client: Optional[MinIOClient] = None


def get_minio_client() -> Optional[MinIOClient]:
    """
    Get or create the global MinIO client instance.
    
    Returns:
        MinIOClient instance if USE_MINIO is enabled, None otherwise
    """
    global _minio_client
    
    if not settings.USE_MINIO:
        return None
    
    if _minio_client is None:
        try:
            _minio_client = MinIOClient()
            # Ensure production bucket exists
            _minio_client.create_bucket(settings.MINIO_PRODUCTION_BUCKET)
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            return None
    
    return _minio_client
