"""
MinIO Helper Utilities for Airflow DAGs.

Provides reusable functions for interacting with MinIO object storage.
"""
import boto3
from botocore.client import Config
from pathlib import Path
from typing import List, Optional
import os


class MinIOClient:
    """Wrapper class for MinIO operations."""
    
    def __init__(
        self,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO endpoint URL (defaults to env var MINIO_ENDPOINT)
            access_key: MinIO access key (defaults to env var AWS_ACCESS_KEY_ID)
            secret_key: MinIO secret key (defaults to env var AWS_SECRET_ACCESS_KEY)
            region: AWS region (default: us-east-1)
        """
        self.endpoint = endpoint or os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
        self.access_key = access_key or os.getenv('AWS_ACCESS_KEY_ID', 'minio_admin')
        self.secret_key = secret_key or os.getenv('AWS_SECRET_ACCESS_KEY', 'minio_password123')
        self.region = region
        
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name=self.region
        )
    
    def list_buckets(self) -> List[str]:
        """List all buckets in MinIO."""
        try:
            response = self.client.list_buckets()
            return [bucket['Name'] for bucket in response['Buckets']]
        except Exception as e:
            print(f"Error listing buckets: {e}")
            return []
    
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        try:
            self.client.head_bucket(Bucket=bucket_name)
            return True
        except:
            return False
    
    def create_bucket(self, bucket_name: str) -> bool:
        """Create a new bucket."""
        try:
            self.client.create_bucket(Bucket=bucket_name)
            print(f"✅ Created bucket: {bucket_name}")
            return True
        except Exception as e:
            print(f"Error creating bucket {bucket_name}: {e}")
            return False
    
    def list_objects(self, bucket_name: str, prefix: str = '') -> List[str]:
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
            print(f"Error listing objects in {bucket_name}/{prefix}: {e}")
            return []
    
    def download_file(self, bucket_name: str, object_key: str, local_path: str) -> bool:
        """
        Download a file from MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_key: Object key (path) in MinIO
            local_path: Local file path to save to
        
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket_name, object_key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading {object_key}: {e}")
            return False
    
    def download_directory(
        self, 
        bucket_name: str, 
        prefix: str, 
        local_dir: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> int:
        """
        Download all files from a MinIO directory to local directory.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Directory prefix in MinIO
            local_dir: Local directory to download to
            exclude_patterns: List of patterns to exclude (e.g., ['*.tmp', '*.log'])
        
        Returns:
            Number of files downloaded
        """
        from fnmatch import fnmatch
        
        exclude_patterns = exclude_patterns or []
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        objects = self.list_objects(bucket_name, prefix)
        download_count = 0
        
        for obj_key in objects:
            # Skip directories
            if obj_key.endswith('/'):
                continue
            
            # Check exclusion patterns
            if any(fnmatch(obj_key, pattern) for pattern in exclude_patterns):
                print(f"⏭️  Skipping excluded: {obj_key}")
                continue
            
            # Calculate local path
            relative_path = obj_key[len(prefix):].lstrip('/')
            local_file = local_dir / relative_path
            
            # Download file
            if self.download_file(bucket_name, obj_key, str(local_file)):
                download_count += 1
                print(f"⬇️  Downloaded: {obj_key} → {local_file}")
        
        return download_count
    
    def upload_file(self, local_path: str, bucket_name: str, object_key: str) -> bool:
        """
        Upload a file to MinIO.
        
        Args:
            local_path: Local file path
            bucket_name: Name of the bucket
            object_key: Object key (path) in MinIO
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.upload_file(local_path, bucket_name, object_key)
            return True
        except Exception as e:
            print(f"Error uploading {local_path} to {object_key}: {e}")
            return False
    
    def upload_directory(
        self, 
        local_dir: str, 
        bucket_name: str, 
        prefix: str = '',
        exclude_patterns: Optional[List[str]] = None
    ) -> int:
        """
        Upload all files from a local directory to MinIO.
        
        Args:
            local_dir: Local directory to upload from
            bucket_name: Name of the bucket
            prefix: Directory prefix in MinIO
            exclude_patterns: List of patterns to exclude (e.g., ['*.tmp', '*.log'])
        
        Returns:
            Number of files uploaded
        """
        from fnmatch import fnmatch
        
        exclude_patterns = exclude_patterns or []
        local_dir = Path(local_dir)
        
        if not local_dir.exists():
            print(f"Error: Directory does not exist: {local_dir}")
            return 0
        
        upload_count = 0
        
        for local_file in local_dir.rglob('*'):
            if local_file.is_file():
                # Check exclusion patterns
                if any(fnmatch(str(local_file.name), pattern) for pattern in exclude_patterns):
                    print(f"⏭️  Skipping excluded: {local_file}")
                    continue
                
                # Calculate MinIO key
                relative_path = local_file.relative_to(local_dir)
                object_key = f"{prefix}/{relative_path}".lstrip('/')
                
                # Upload file
                if self.upload_file(str(local_file), bucket_name, object_key):
                    upload_count += 1
                    print(f"⬆️  Uploaded: {local_file} → s3://{bucket_name}/{object_key}")
        
        return upload_count
    
    def delete_object(self, bucket_name: str, object_key: str) -> bool:
        """Delete an object from MinIO."""
        try:
            self.client.delete_object(Bucket=bucket_name, Key=object_key)
            return True
        except Exception as e:
            print(f"Error deleting {object_key}: {e}")
            return False
    
    def get_object_metadata(self, bucket_name: str, object_key: str) -> Optional[dict]:
        """Get metadata for an object."""
        try:
            response = self.client.head_object(Bucket=bucket_name, Key=object_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType'),
                'etag': response.get('ETag')
            }
        except Exception as e:
            print(f"Error getting metadata for {object_key}: {e}")
            return None


def download_dataset_from_minio(
    bucket_name: str = 'training-data',
    local_dir: str = '/tmp/training_data',
    minio_client: Optional[MinIOClient] = None
) -> str:
    """
    Download complete dataset from MinIO.
    
    Args:
        bucket_name: MinIO bucket name
        local_dir: Local directory to download to
        minio_client: Optional MinIOClient instance (will create if not provided)
    
    Returns:
        Path to downloaded dataset directory
    """
    if minio_client is None:
        minio_client = MinIOClient()
    
    print(f"📥 Downloading dataset from MinIO bucket: {bucket_name}")
    
    # Download entire bucket contents
    count = minio_client.download_directory(
        bucket_name=bucket_name,
        prefix='',
        local_dir=local_dir,
        exclude_patterns=['*.tmp', '*.log', '.DS_Store']
    )
    
    print(f"✅ Downloaded {count} files to {local_dir}")
    
    return local_dir


def upload_model_to_minio(
    model_path: str,
    model_name: str,
    version: str,
    bucket_name: str = 'model-exports',
    minio_client: Optional[MinIOClient] = None
) -> bool:
    """
    Upload trained model to MinIO.
    
    Args:
        model_path: Local path to model file
        model_name: Name of the model
        version: Model version
        bucket_name: MinIO bucket name
        minio_client: Optional MinIOClient instance
    
    Returns:
        True if successful, False otherwise
    """
    if minio_client is None:
        minio_client = MinIOClient()
    
    object_key = f"{model_name}/v{version}/model.pt"
    
    print(f"📤 Uploading model to MinIO: s3://{bucket_name}/{object_key}")
    
    success = minio_client.upload_file(
        local_path=model_path,
        bucket_name=bucket_name,
        object_key=object_key
    )
    
    if success:
        print(f"✅ Model uploaded successfully")
    else:
        print(f"❌ Failed to upload model")
    
    return success


def sync_production_data_to_minio(
    local_dir: str,
    bucket_name: str = 'production-data',
    date_prefix: str = None,
    minio_client: Optional[MinIOClient] = None
) -> int:
    """
    Sync production inference data to MinIO.
    
    Args:
        local_dir: Local directory containing production data
        bucket_name: MinIO bucket name
        date_prefix: Optional date prefix for organization (e.g., '2025/02/10')
        minio_client: Optional MinIOClient instance
    
    Returns:
        Number of files uploaded
    """
    from datetime import datetime
    
    if minio_client is None:
        minio_client = MinIOClient()
    
    if date_prefix is None:
        date_prefix = datetime.now().strftime('%Y/%m/%d')
    
    print(f"🔄 Syncing production data to MinIO: {local_dir} → s3://{bucket_name}/{date_prefix}")
    
    count = minio_client.upload_directory(
        local_dir=local_dir,
        bucket_name=bucket_name,
        prefix=date_prefix,
        exclude_patterns=['*.tmp', '*.log', '.DS_Store']
    )
    
    print(f"✅ Synced {count} files to MinIO")
    
    return count
