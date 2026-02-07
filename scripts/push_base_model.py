"""
Script to push all YOLOv26 base models to MinIO using AWS SDK (boto3)
Downloads models from Ultralytics if not present locally
Uploads yolo26n, yolo26s, yolo26m, yolo26l, and yolo26x models
"""
import os
import sys
from pathlib import Path
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import requests
from tqdm import tqdm
from dotenv import load_dotenv


def get_minio_client():
    """
    Create and return a MinIO client using boto3 S3 client
    """
    endpoint_url = os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    
    print(f"Connecting to MinIO at: {endpoint_url}")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    return s3_client


def create_bucket_if_not_exists(s3_client, bucket_name):
    """
    Create a bucket if it doesn't already exist
    """
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Bucket '{bucket_name}' already exists")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Bucket doesn't exist, create it
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"✓ Created bucket '{bucket_name}'")
                return True
            except ClientError as create_error:
                print(f"✗ Error creating bucket: {create_error}")
                return False
        else:
            print(f"✗ Error checking bucket: {e}")
            return False


def upload_file_to_minio(s3_client, file_path, bucket_name, object_name=None):
    """
    Upload a file to MinIO bucket
    
    Args:
        s3_client: boto3 S3 client
        file_path: Path to file to upload
        bucket_name: Name of the bucket
        object_name: S3 object name. If not specified, file_path basename is used
    """
    if object_name is None:
        object_name = Path(file_path).name
    
    try:
        file_size = os.path.getsize(file_path)
        print(f"Uploading {file_path} ({file_size / 1024 / 1024:.2f} MB) to {bucket_name}/{object_name}...")
        
        s3_client.upload_file(
            file_path,
            bucket_name,
            object_name,
            ExtraArgs={'ContentType': 'application/octet-stream'}
        )
        
        print(f"✓ Successfully uploaded {object_name} to bucket {bucket_name}")
        return True
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return False
    except ClientError as e:
        print(f"✗ Error uploading file: {e}")
        return False


def list_bucket_contents(s3_client, bucket_name, prefix=''):
    """
    List contents of a bucket
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' in response:
            print(f"\nContents of bucket '{bucket_name}':")
            for obj in response['Contents']:
                size_mb = obj['Size'] / 1024 / 1024
                print(f"  - {obj['Key']} ({size_mb:.2f} MB)")
        else:
            print(f"\nBucket '{bucket_name}' is empty")
    except ClientError as e:
        print(f"✗ Error listing bucket contents: {e}")


def download_yolo26_model(model_name, save_path):
    """
    Download YOLOv26 model from Ultralytics GitHub releases
    
    Args:
        model_name: Name of the model (e.g., 'yolo26n.pt', 'yolo26s.pt')
        save_path: Path where to save the downloaded model
    """
    # Map model names to download URLs
    model_urls = {
        'yolo26n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt',
        'yolo26s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt',
        'yolo26m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt',
        'yolo26l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt',
        'yolo26x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt',
    }
    
    if model_name not in model_urls:
        print(f"✗ Unknown model: {model_name}")
        print(f"Available models: {', '.join(model_urls.keys())}")
        return False
    
    url = model_urls[model_name]
    
    try:
        print(f"Downloading {model_name} from {url}...")
        
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=model_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Successfully downloaded {model_name} to {save_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading model: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False
    except Exception as e:
        print(f"✗ Unexpected error during download: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def ensure_model_exists(model_path):
    """
    Ensure the model file exists, download if necessary
    
    Args:
        model_path: Path object pointing to the model file
    
    Returns:
        bool: True if model exists or was successfully downloaded
    """
    if model_path.exists():
        file_size = model_path.stat().st_size / 1024 / 1024
        print(f"✓ Model file found: {model_path} ({file_size:.2f} MB)")
        return True
    
    print(f"⚠ Model file not found: {model_path}")
    model_name = model_path.name
    
    # Try to download the model
    print(f"Attempting to download {model_name}...")
    return download_yolo26_model(model_name, str(model_path))


def main():
    """
    Main function to push all YOLOv26 base models to MinIO
    """
    # Load environment variables from .env file
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        print(f"⚠ No .env file found at {env_path}, using defaults or system environment variables")
    
    # Configuration
    bucket_name = os.getenv("MODEL_BUCKET", "models")
    
    # All YOLOv26 models to upload
    models_to_upload = [
        'yolo26n.pt',
        'yolo26s.pt',
        'yolo26m.pt',
        'yolo26l.pt',
        'yolo26x.pt'
    ]
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("MinIO Model Upload Script - All YOLOv26 Models")
    print("=" * 60)
    print(f"Bucket: {bucket_name}")
    print(f"Models to upload: {', '.join(models_to_upload)}")
    print("=" * 60)
    
    try:
        # Create MinIO client
        s3_client = get_minio_client()
        
        # Create bucket if it doesn't exist
        if not create_bucket_if_not_exists(s3_client, bucket_name):
            sys.exit(1)
        
        # Track upload status
        successful_uploads = []
        failed_uploads = []
        
        # Upload each model
        for model_name in models_to_upload:
            print(f"\n{'=' * 60}")
            print(f"Processing: {model_name}")
            print(f"{'=' * 60}")
            
            full_model_path = project_root / model_name
            object_name = f"base/{model_name}"
            
            # Ensure model file exists (download if necessary)
            if not ensure_model_exists(full_model_path):
                print(f"✗ Error: Could not obtain model file at {full_model_path}")
                failed_uploads.append(model_name)
                continue
            
            # Upload model file
            if upload_file_to_minio(s3_client, str(full_model_path), bucket_name, object_name):
                successful_uploads.append(model_name)
            else:
                failed_uploads.append(model_name)
        
        # List bucket contents
        print(f"\n{'=' * 60}")
        list_bucket_contents(s3_client, bucket_name)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Upload Summary")
        print("=" * 60)
        print(f"✓ Successfully uploaded: {len(successful_uploads)}/{len(models_to_upload)} models")
        if successful_uploads:
            for model in successful_uploads:
                print(f"  - {model}")
        
        if failed_uploads:
            print(f"\n✗ Failed uploads: {len(failed_uploads)} models")
            for model in failed_uploads:
                print(f"  - {model}")
        
        print("=" * 60)
        
        # Exit with error if any uploads failed
        if failed_uploads:
            sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
