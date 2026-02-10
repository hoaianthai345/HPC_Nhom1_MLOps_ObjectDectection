"""
Script to delete MinIO bucket and its contents using AWS SDK (boto3)
"""
import os
import sys
from pathlib import Path
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv


def get_minio_client():
    """
    Create and return a MinIO client using boto3 S3 client
    """
    endpoint_url = os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minio_admin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minio_password123")
    
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


def delete_all_objects(s3_client, bucket_name, prefix=None):
    """
    Delete all objects in a bucket or with a specific prefix (folder)
    
    Args:
        s3_client: boto3 S3 client
        bucket_name: Name of the bucket
        prefix: Optional prefix/folder path (e.g., 'models/yolo/')
    """
    try:
        target = f"folder '{prefix}' in bucket '{bucket_name}'" if prefix else f"bucket '{bucket_name}'"
        print(f"Listing objects in {target}...")
        
        # List objects with optional prefix
        list_params = {'Bucket': bucket_name}
        if prefix:
            list_params['Prefix'] = prefix
        
        response = s3_client.list_objects_v2(**list_params)
        
        if 'Contents' not in response:
            print(f"✓ {target.capitalize()} is already empty or does not exist")
            return True
        
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
        
        print(f"Deleting {len(objects_to_delete)} object(s)...")
        for obj in response['Contents']:
            print(f"  - Deleting {obj['Key']}")
        
        # Delete in batches of 1000 (S3 limit)
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i:i+1000]
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': batch}
            )
        
        print(f"✓ Successfully deleted all objects from {target}")
        return True
        
    except ClientError as e:
        print(f"✗ Error deleting objects: {e}")
        return False


def delete_bucket(s3_client, bucket_name):
    """
    Delete a bucket (must be empty first)
    """
    try:
        s3_client.delete_bucket(Bucket=bucket_name)
        print(f"✓ Successfully deleted bucket '{bucket_name}'")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            print(f"✓ Bucket '{bucket_name}' does not exist")
            return True
        else:
            print(f"✗ Error deleting bucket: {e}")
            return False


def main():
    """
    Main function to delete bucket/folder and its contents
    
    Usage:
        python delete_bucket.py <bucket_name>                    # Delete entire bucket
        python delete_bucket.py <bucket_name> <folder_path>      # Delete folder in bucket
    
    Examples:
        python delete_bucket.py base-models                      # Delete entire bucket
        python delete_bucket.py base-models models/yolo/         # Delete folder in bucket
    """
    # Load environment variables from .env file
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    
    # Parse command line arguments
    bucket_name = None
    folder_path = None
    
    if len(sys.argv) > 1:
        bucket_name = sys.argv[1]
        if len(sys.argv) > 2:
            folder_path = sys.argv[2]
            # Ensure folder path doesn't start with /
            if folder_path.startswith('/'):
                folder_path = folder_path[1:]
    else:
        bucket_name = os.getenv("MODEL_BUCKET", "base-models")
    
    print("=" * 60)
    print("MinIO Bucket/Folder Deletion Script")
    print("=" * 60)
    print(f"Bucket: {bucket_name}")
    if folder_path:
        print(f"Folder: {folder_path}")
        print("Action: Delete folder only")
    else:
        print("Action: Delete entire bucket")
    print("=" * 60)
    
    # Ask for confirmation
    if folder_path:
        target = f"folder '{folder_path}' in bucket '{bucket_name}'"
        action = "delete this folder"
    else:
        target = f"bucket '{bucket_name}' and all its contents"
        action = "delete this entire bucket"
    
    confirm = input(f"\n⚠️  Are you sure you want to {action}? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("Deletion cancelled.")
        sys.exit(0)
    
    try:
        # Create MinIO client
        s3_client = get_minio_client()
        
        if folder_path:
            # Delete only objects with the specified prefix (folder)
            if not delete_all_objects(s3_client, bucket_name, prefix=folder_path):
                sys.exit(1)
            print("\n" + "=" * 60)
            print(f"✓ Folder '{folder_path}' deletion completed successfully!")
            print("=" * 60)
        else:
            # Delete all objects in the bucket
            if not delete_all_objects(s3_client, bucket_name):
                sys.exit(1)
            
            # Delete the bucket itself
            if not delete_bucket(s3_client, bucket_name):
                sys.exit(1)
            
            print("\n" + "=" * 60)
            print("✓ Bucket deletion completed successfully!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
