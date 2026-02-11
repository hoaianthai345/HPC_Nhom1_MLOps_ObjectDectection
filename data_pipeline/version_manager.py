"""
Data Version Manager for MLOps Training Pipeline

Manages versioned training datasets in MinIO with metadata tracking.
Refactored for consistency and integration with data_pipeline module.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import json
from tqdm import tqdm
import hashlib

from .config import config as pipeline_config


class DataVersionManager:
    """
    Manages versioned training data in MinIO
    """
    
    def __init__(self, bucket_name=None):
        if bucket_name is None:
            bucket_name = pipeline_config.training_bucket
        
        self.bucket_name = bucket_name
        self.endpoint_url = pipeline_config.minio_endpoint
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=pipeline_config.minio_access_key,
            aws_secret_access_key=pipeline_config.minio_secret_key,
            config=BotoConfig(signature_version='s3v4'),
            region_name=pipeline_config.minio_region
        )
        
        print(f"🔗 Connected to MinIO at: {self.endpoint_url}")
        print(f"📦 Bucket: {self.bucket_name}")
    
    def compute_file_hash(self, file_path):
        """
        Compute SHA256 hash of file
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def list_versions(self):
        """
        List all data versions in bucket
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Delimiter='/'
            )
            
            versions = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    version = prefix['Prefix'].rstrip('/')
                    versions.append(version)
            
            return sorted(versions, reverse=True)
        except ClientError as e:
            print(f"✗ Error listing versions: {e}")
            return []
    
    def get_version_info(self, version):
        """
        Get metadata for a specific version
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"{version}/manifest.json"
            )
            manifest = json.loads(response['Body'].read().decode('utf-8'))
            return manifest
        except ClientError:
            return None
    
    def get_latest_version(self):
        """
        Get the latest version number
        """
        versions = self.list_versions()
        if versions:
            return versions[0]
        return None
    
    def create_new_version(self, version=None, description="", metadata=None):
        """
        Create a new version with automatic version numbering
        """
        if version is None:
            # Auto-generate version
            latest = self.get_latest_version()
            if latest:
                # Extract version number and increment
                try:
                    major, minor = latest.split('v')[1].split('.')
                    new_minor = int(minor) + 1
                    version = f"v{major}.{new_minor}"
                except:
                    version = f"v1.{len(self.list_versions()) + 1}"
            else:
                version = "v1.0"
        
        print(f"\n📝 Creating new version: {version}")
        
        # Create version manifest
        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "metadata": metadata or {},
            "splits": ["train", "valid", "test"]
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"{version}/manifest.json",
                Body=json.dumps(manifest, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
            print(f"✓ Version {version} created")
            return version
        except ClientError as e:
            print(f"✗ Error creating version: {e}")
            return None
    
    def upload_dataset(self, local_dir, version=None, description="", metadata=None):
        """
        Upload a complete dataset as a new version
        """
        local_path = Path(local_dir)
        if not local_path.exists():
            print(f"✗ Directory not found: {local_dir}")
            return False
        
        # Create new version
        if version is None:
            version = self.create_new_version(description=description, metadata=metadata)
        else:
            self.create_new_version(version=version, description=description, metadata=metadata)
        
        if not version:
            return False
        
        print(f"\n📤 Uploading dataset to {version}...")
        
        # Get all files
        all_files = list(local_path.rglob('*'))
        files_to_upload = [f for f in all_files if f.is_file()]
        
        success_count = 0
        failed_count = 0
        
        with tqdm(total=len(files_to_upload), desc="Uploading") as pbar:
            for file_path in files_to_upload:
                relative_path = file_path.relative_to(local_path)
                object_name = f"{version}/{relative_path}"
                
                try:
                    self.s3_client.upload_file(
                        str(file_path),
                        self.bucket_name,
                        object_name,
                        ExtraArgs={'ContentType': 'application/octet-stream'}
                    )
                    success_count += 1
                except Exception as e:
                    print(f"\n✗ Error uploading {file_path}: {e}")
                    failed_count += 1
                
                pbar.update(1)
        
        print(f"\n✓ Upload complete: {success_count} succeeded, {failed_count} failed")
        
        # Update manifest with file statistics
        self.update_version_stats(version)
        
        return failed_count == 0
    
    def update_version_stats(self, version):
        """
        Update version manifest with dataset statistics
        """
        try:
            # Count files in each split
            stats = {}
            for split in ['train', 'valid', 'test']:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"{version}/{split}/"
                )
                
                images = 0
                labels = 0
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        if '/images/' in key:
                            images += 1
                        elif '/labels/' in key:
                            labels += 1
                
                stats[split] = {
                    "images": images,
                    "labels": labels
                }
            
            # Get and update manifest
            manifest = self.get_version_info(version)
            if manifest:
                manifest['stats'] = stats
                manifest['updated_at'] = datetime.now().isoformat()
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f"{version}/manifest.json",
                    Body=json.dumps(manifest, indent=2).encode('utf-8'),
                    ContentType='application/json'
                )
                
                print(f"✓ Statistics updated for {version}")
            
        except ClientError as e:
            print(f"⚠ Warning: Could not update stats: {e}")
    
    def download_version(self, version, local_dir):
        """
        Download a specific version to local directory
        """
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📥 Downloading {version} to {local_dir}...")
        
        try:
            # List all objects in version
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{version}/"
            )
            
            if 'Contents' not in response:
                print(f"✗ No files found for version {version}")
                return False
            
            files = response['Contents']
            
            with tqdm(total=len(files), desc="Downloading") as pbar:
                for obj in files:
                    key = obj['Key']
                    # Remove version prefix
                    relative_path = key[len(f"{version}/"):]
                    
                    if not relative_path or relative_path.endswith('/'):
                        pbar.update(1)
                        continue
                    
                    local_file = local_path / relative_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.s3_client.download_file(
                        self.bucket_name,
                        key,
                        str(local_file)
                    )
                    
                    pbar.update(1)
            
            print(f"\n✓ Download complete")
            return True
            
        except ClientError as e:
            print(f"✗ Error downloading version: {e}")
            return False
    
    def delete_version(self, version):
        """
        Delete a specific version
        """
        print(f"🗑️  Deleting version {version}...")
        
        try:
            # List all objects in version
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{version}/"
            )
            
            if 'Contents' not in response:
                print(f"✗ Version {version} not found")
                return False
            
            # Delete all objects
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
            
            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': objects_to_delete}
            )
            
            print(f"✓ Version {version} deleted")
            return True
            
        except ClientError as e:
            print(f"✗ Error deleting version: {e}")
            return False
    
    def compare_versions(self, version1, version2):
        """
        Compare two versions
        """
        print(f"\n📊 Comparing {version1} vs {version2}\n")
        
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        if not info1 or not info2:
            print("✗ Could not load version information")
            return
        
        print(f"Version: {version1}")
        print(f"  Created: {info1.get('created_at', 'Unknown')}")
        print(f"  Description: {info1.get('description', 'N/A')}")
        if 'stats' in info1:
            for split, stats in info1['stats'].items():
                print(f"  {split}: {stats['images']} images, {stats['labels']} labels")
        
        print(f"\nVersion: {version2}")
        print(f"  Created: {info2.get('created_at', 'Unknown')}")
        print(f"  Description: {info2.get('description', 'N/A')}")
        if 'stats' in info2:
            for split, stats in info2['stats'].items():
                print(f"  {split}: {stats['images']} images, {stats['labels']} labels")
    
    def show_version_tree(self):
        """
        Display version tree with information
        """
        versions = self.list_versions()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           📚 Data Version Tree                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        if not versions:
            print("  No versions found")
            return
        
        for i, version in enumerate(versions):
            info = self.get_version_info(version)
            
            prefix = "└─" if i == len(versions) - 1 else "├─"
            print(f"{prefix} {version}")
            
            if info:
                print(f"   ├─ Created: {info.get('created_at', 'Unknown')}")
                print(f"   ├─ Description: {info.get('description', 'N/A')}")
                
                if 'stats' in info:
                    print(f"   └─ Stats:")
                    for split, stats in info['stats'].items():
                        print(f"      • {split}: {stats['images']} images, {stats['labels']} labels")
                print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage training data versions in MinIO")
    parser.add_argument('--bucket', default='training-data', help='Bucket name')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List versions
    subparsers.add_parser('list', help='List all versions')
    
    # Show version tree
    subparsers.add_parser('tree', help='Show version tree')
    
    # Upload dataset
    upload_parser = subparsers.add_parser('upload', help='Upload dataset as new version')
    upload_parser.add_argument('--dir', required=True, help='Local directory path')
    upload_parser.add_argument('--version', help='Version name (auto if not specified)')
    upload_parser.add_argument('--description', default='', help='Version description')
    
    # Download version
    download_parser = subparsers.add_parser('download', help='Download version')
    download_parser.add_argument('--version', required=True, help='Version to download')
    download_parser.add_argument('--dir', required=True, help='Local directory path')
    
    # Delete version
    delete_parser = subparsers.add_parser('delete', help='Delete version')
    delete_parser.add_argument('--version', required=True, help='Version to delete')
    
    # Compare versions
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('--v1', required=True, help='First version')
    compare_parser.add_argument('--v2', required=True, help='Second version')
    
    # Info about version
    info_parser = subparsers.add_parser('info', help='Show version information')
    info_parser.add_argument('--version', required=True, help='Version name')
    
    args = parser.parse_args()
    
    manager = DataVersionManager(bucket_name=args.bucket)
    
    if args.command == 'list':
        versions = manager.list_versions()
        print(f"\n📋 Available versions ({len(versions)}):\n")
        for v in versions:
            print(f"  • {v}")
    
    elif args.command == 'tree':
        manager.show_version_tree()
    
    elif args.command == 'upload':
        metadata = {
            "source": "local",
            "uploaded_by": os.getenv("USER", "unknown")
        }
        manager.upload_dataset(
            args.dir,
            version=args.version,
            description=args.description,
            metadata=metadata
        )
    
    elif args.command == 'download':
        manager.download_version(args.version, args.dir)
    
    elif args.command == 'delete':
        confirm = input(f"⚠️  Delete version {args.version}? (yes/no): ")
        if confirm.lower() == 'yes':
            manager.delete_version(args.version)
    
    elif args.command == 'compare':
        manager.compare_versions(args.v1, args.v2)
    
    elif args.command == 'info':
        info = manager.get_version_info(args.version)
        if info:
            print(f"\n📄 Version: {args.version}\n")
            print(json.dumps(info, indent=2))
        else:
            print(f"✗ Version {args.version} not found")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
