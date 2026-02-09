"""
Script to push base training data to MinIO using AWS SDK (boto3)
Uploads YOLO dataset (images and labels) to MinIO buckets for training.
"""
import os
import sys
from pathlib import Path
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from tqdm import tqdm
from dotenv import load_dotenv
import yaml


def get_minio_client():
    """
    Create and return a MinIO client using boto3 S3 client
    """
    load_dotenv()
    
    endpoint_url = os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    
    print(f"🔗 Connecting to MinIO at: {endpoint_url}")
    
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
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Bucket '{bucket_name}' already exists")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
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
    """
    if object_name is None:
        object_name = str(file_path)
    
    try:
        s3_client.upload_file(
            str(file_path),
            bucket_name,
            object_name,
            ExtraArgs={'ContentType': 'application/octet-stream'}
        )
        return True
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return False
    except ClientError as e:
        print(f"✗ Error uploading file: {e}")
        return False


def upload_directory_to_minio(s3_client, local_dir, bucket_name, prefix=''):
    """
    Upload entire directory to MinIO bucket with progress bar
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"✗ Directory not found: {local_dir}")
        return False
    
    # Get all files
    all_files = list(local_path.rglob('*'))
    files_to_upload = [f for f in all_files if f.is_file()]
    
    print(f"\n📤 Uploading {len(files_to_upload)} files from {local_dir}...")
    
    success_count = 0
    failed_count = 0
    
    with tqdm(total=len(files_to_upload), desc="Uploading") as pbar:
        for file_path in files_to_upload:
            # Create object name preserving directory structure
            relative_path = file_path.relative_to(local_path)
            object_name = f"{prefix}/{relative_path}" if prefix else str(relative_path)
            
            if upload_file_to_minio(s3_client, file_path, bucket_name, object_name):
                success_count += 1
            else:
                failed_count += 1
            
            pbar.update(1)
    
    print(f"\n✓ Upload complete: {success_count} succeeded, {failed_count} failed")
    return failed_count == 0


def upload_dataset(data_dir: str, bucket_name: str = "yolo-dataset"):
    """
    Upload complete YOLO dataset to MinIO
    Uploads train, valid, and test splits including images and labels
    """
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           📦 Push Base Training Data to MinIO                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    s3_client = get_minio_client()
    
    # Create bucket
    if not create_bucket_if_not_exists(s3_client, bucket_name):
        print("✗ Failed to create/access bucket")
        return False
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    # Upload data.yaml config file
    data_yaml = data_path / "data.yaml"
    if data_yaml.exists():
        print("\n📄 Uploading data.yaml...")
        upload_file_to_minio(s3_client, data_yaml, bucket_name, "data.yaml")
    
    # Upload each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_dir = data_path / split
        if split_dir.exists():
            print(f"\n📁 Processing {split} split...")
            
            # Upload images
            images_dir = split_dir / "images"
            if images_dir.exists():
                upload_directory_to_minio(
                    s3_client, 
                    images_dir, 
                    bucket_name, 
                    prefix=f"{split}/images"
                )
            
            # Upload labels
            labels_dir = split_dir / "labels"
            if labels_dir.exists():
                upload_directory_to_minio(
                    s3_client, 
                    labels_dir, 
                    bucket_name, 
                    prefix=f"{split}/labels"
                )
        else:
            print(f"⚠ Split directory not found: {split_dir}")
    
    print("\n" + "="*70)
    print("✓ Dataset upload completed successfully!")
    print("="*70)
    return True


def list_bucket_contents(s3_client, bucket_name, prefix='', max_keys=100):
    """
    List contents of a bucket
    """
    try:
        print(f"\n📋 Listing contents of bucket '{bucket_name}' (prefix: '{prefix}'):")
        print("-" * 70)
        
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=prefix,
            PaginationConfig={'MaxItems': max_keys}
        )
        
        total_size = 0
        file_count = 0
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    size_mb = obj['Size'] / (1024 * 1024)
                    print(f"  {obj['Key']:<50} {size_mb:>8.2f} MB")
                    total_size += obj['Size']
                    file_count += 1
        
        print("-" * 70)
        print(f"Total: {file_count} files, {total_size / (1024 * 1024):.2f} MB")
        return True
        
    except ClientError as e:
        print(f"✗ Error listing bucket: {e}")
        return False


def verify_upload(bucket_name: str = "yolo-dataset"):
    """
    Verify uploaded dataset structure
    """
    print("\n🔍 Verifying upload...")
    s3_client = get_minio_client()
    
    # Check each split
    for split in ['train', 'valid', 'test']:
        print(f"\n📂 {split.upper()} split:")
        list_bucket_contents(s3_client, bucket_name, prefix=f"{split}/", max_keys=10)


def main():
    """
    Main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Push base training data to MinIO")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data_final",
        help="Path to dataset directory (default: data_final)"
    )
    parser.add_argument(
        "--bucket", 
        type=str, 
        default="yolo-dataset",
        help="MinIO bucket name (default: yolo-dataset)"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify upload after completion"
    )
    parser.add_argument(
        "--list-only", 
        action="store_true",
        help="Only list bucket contents without uploading"
    )
    
    args = parser.parse_args()
    
    try:
        if args.list_only:
            verify_upload(args.bucket)
        else:
            success = upload_dataset(args.data_dir, args.bucket)
            
            if success and args.verify:
                verify_upload(args.bucket)
            
            sys.exit(0 if success else 1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
