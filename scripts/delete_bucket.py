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


def delete_all_objects(s3_client, bucket_name):
    """
    Delete all objects in a bucket
    """
    try:
        print(f"Listing objects in bucket '{bucket_name}'...")
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' not in response:
            print(f"✓ Bucket '{bucket_name}' is already empty")
            return True
        
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
        
        print(f"Deleting {len(objects_to_delete)} object(s)...")
        for obj in response['Contents']:
            print(f"  - Deleting {obj['Key']}")
        
        s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={'Objects': objects_to_delete}
        )
        
        print(f"✓ Successfully deleted all objects from bucket '{bucket_name}'")
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
    Main function to delete bucket and its contents
    """
    # Load environment variables from .env file
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    
    # Get bucket name from environment or command line
    if len(sys.argv) > 1:
        bucket_name = sys.argv[1]
    else:
        bucket_name = os.getenv("MODEL_BUCKET", "base-models")
    
    print("=" * 60)
    print("MinIO Bucket Deletion Script")
    print("=" * 60)
    print(f"Bucket to delete: {bucket_name}")
    print("=" * 60)
    
    # Ask for confirmation
    confirm = input(f"\n⚠️  Are you sure you want to delete bucket '{bucket_name}' and all its contents? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("Deletion cancelled.")
        sys.exit(0)
    
    try:
        # Create MinIO client
        s3_client = get_minio_client()
        
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
