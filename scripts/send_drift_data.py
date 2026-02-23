"""
Script to send test variant images to the serving API for drift detection.
This will send images from data/test_variants directories to trigger data drift.

Additionally, this script can verify that images are properly saved to the
production-data bucket (MinIO or local filesystem) for later teacher model inference.

Usage:
    # Send test variant images for drift detection (CPU)
    python scripts/send_drift_data.py --test-dir data/test_variants
    
    # Send to GPU endpoint for faster inference (auto-switches to port 8001)
    python scripts/send_drift_data.py --test-dir data/test_variants --use-gpu
    
    # Send with MinIO verification
    python scripts/send_drift_data.py --test-dir data/test_variants --verify-minio
    
    # Check production-data bucket status
    python scripts/send_drift_data.py --check-bucket --minio-endpoint http://localhost:9000
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DriftDataSender:
    """Send test variant images to detection API and verify production data storage."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        delay: float = 0.0,
        use_gpu: bool = False,
        verify_minio: bool = False,
        minio_endpoint: Optional[str] = None,
        minio_access_key: Optional[str] = None,
        minio_secret_key: Optional[str] = None,
    ):
        """
        Initialize the drift data sender.
        
        Args:
            api_url: Base URL of the serving API
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            delay: Delay between requests in seconds
            use_gpu: Use GPU endpoint (/detect-gpu) instead of CPU (/detect)
            verify_minio: Whether to verify images are saved to MinIO
            minio_endpoint: MinIO endpoint URL
            minio_access_key: MinIO access key
            minio_secret_key: MinIO secret key
        """
        self.api_url = api_url.rstrip("/")
        self.use_gpu = use_gpu
        self.detect_endpoint = f"{self.api_url}/detect-gpu" if use_gpu else f"{self.api_url}/detect"
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.delay = delay
        self.results: List[Dict[str, Any]] = []
        self.verify_minio = verify_minio
        self.minio_client = None
        
        # Initialize MinIO client if verification requested
        if verify_minio:
            self._init_minio_client(
                minio_endpoint or "http://localhost:9000",
                minio_access_key or "minio_admin",
                minio_secret_key or "minio_password123",
            )
    
    def _init_minio_client(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        """Initialize MinIO client for verification."""
        try:
            import boto3
            from botocore.client import Config
            
            self.minio_client = boto3.client(
                's3',
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version='s3v4'),
                region_name='us-east-1'
            )
            logger.info(f"✓ MinIO client initialized (endpoint: {endpoint})")
        except ImportError:
            logger.warning("boto3 not installed, MinIO verification disabled")
            self.verify_minio = False
        except Exception as e:
            logger.warning(f"Failed to initialize MinIO client: {e}")
            self.verify_minio = False
    
    def verify_production_data_storage(
        self,
        bucket_name: str = "production-data",
    ) -> Dict[str, Any]:
        """
        Verify that images are being saved to production-data bucket.
        
        Args:
            bucket_name: Name of the production data bucket
            
        Returns:
            Dictionary with verification results
        """
        if not self.verify_minio or not self.minio_client:
            return {"verified": False, "reason": "MinIO verification not enabled"}
        
        try:
            # Get today's date folder
            today = datetime.now().strftime("%Y-%m-%d")
            
            # List objects in today's folder
            paginator = self.minio_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=f"{today}/")
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    objects.extend([obj['Key'] for obj in page['Contents']])
            
            return {
                "verified": True,
                "bucket": bucket_name,
                "date": today,
                "image_count": len(objects),
                "objects": objects[:10],  # Show first 10
            }
        except Exception as e:
            logger.error(f"Failed to verify MinIO storage: {e}")
            return {"verified": False, "error": str(e)}
        
    def find_images(self, root_dir: Path) -> Dict[str, List[Path]]:
        """
        Find all images in test variant directories.
        
        Args:
            root_dir: Root directory containing test variant folders
            
        Returns:
            Dictionary mapping variant name to list of image paths
        """
        variants = {}
        
        if not root_dir.exists():
            logger.error(f"Directory not found: {root_dir}")
            return variants
        
        # Find all subdirectories
        for variant_dir in root_dir.iterdir():
            if not variant_dir.is_dir():
                continue
                
            variant_name = variant_dir.name
            images_dir = variant_dir / "images"
            
            if not images_dir.exists():
                logger.warning(f"No images directory found in {variant_dir}")
                continue
            
            # Find all image files
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            images = [
                img for img in images_dir.iterdir()
                if img.is_file() and img.suffix.lower() in image_extensions
            ]
            
            if images:
                variants[variant_name] = sorted(images)
                logger.info(f"Found {len(images)} images in {variant_name}")
        
        return variants
    
    def send_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Send a single image to the detection API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing response data and metadata
        """
        try:
            # Prepare the request
            with open(image_path, "rb") as f:
                files = {"file": (image_path.name, f, "image/jpeg")}
                params = {
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold,
                }
                
                # Send request (increased timeout for GPU inference + MinIO operations)
                response = requests.post(
                    self.detect_endpoint,
                    files=files,
                    params=params,
                    timeout=120,
                )
                
            result = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "status_code": response.status_code,
                "success": response.status_code == 200,
            }
            
            if response.status_code == 200:
                data = response.json()
                result.update({
                    "num_detections": data.get("num_detections", 0),
                    "inference_time_ms": data.get("inference_time_ms", 0),
                    "request_id": data.get("request_id", ""),
                })
                logger.debug(
                    f"Successfully processed {image_path.name}: "
                    f"{data.get('num_detections', 0)} detections"
                )
            else:
                result["error"] = response.text
                logger.warning(
                    f"Failed to process {image_path.name}: "
                    f"status={response.status_code}"
                )
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {image_path.name}: {e}")
            return {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "status_code": 0,
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Unexpected error for {image_path.name}: {e}")
            return {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "status_code": 0,
                "success": False,
                "error": str(e),
            }
    
    def send_batch(
        self,
        variant_name: str,
        images: List[Path],
        max_images: int = None,
    ) -> Dict[str, Any]:
        """
        Send a batch of images from a variant.
        
        Args:
            variant_name: Name of the test variant
            images: List of image paths
            max_images: Maximum number of images to send (None = all)
            
        Returns:
            Summary statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing variant: {variant_name}")
        logger.info(f"{'='*60}")
        
        images_to_send = images[:max_images] if max_images else images
        logger.info(f"Sending {len(images_to_send)} images...")
        
        success_count = 0
        error_count = 0
        total_detections = 0
        total_inference_time = 0.0
        
        for image_path in tqdm(images_to_send, desc=f"{variant_name}"):
            result = self.send_image(image_path)
            self.results.append({"variant": variant_name, **result})
            
            if result["success"]:
                success_count += 1
                total_detections += result.get("num_detections", 0)
                total_inference_time += result.get("inference_time_ms", 0)
            else:
                error_count += 1
            
            # Delay between requests if specified
            if self.delay > 0:
                time.sleep(self.delay)
        
        summary = {
            "variant": variant_name,
            "total_images": len(images_to_send),
            "success": success_count,
            "errors": error_count,
            "total_detections": total_detections,
            "avg_inference_time_ms": (
                total_inference_time / success_count if success_count > 0 else 0
            ),
        }
        
        logger.info(f"\nSummary for {variant_name}:")
        logger.info(f"  Total images: {summary['total_images']}")
        logger.info(f"  Success: {summary['success']}")
        logger.info(f"  Errors: {summary['errors']}")
        logger.info(f"  Total detections: {summary['total_detections']}")
        logger.info(f"  Avg inference time: {summary['avg_inference_time_ms']:.2f} ms")
        
        return summary
    
    def run(
        self,
        test_variants_dir: Path,
        variants: List[str] = None,
        max_images_per_variant: int = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the drift data sending process.
        
        Args:
            test_variants_dir: Directory containing test variant folders
            variants: List of specific variants to process (None = all)
            max_images_per_variant: Maximum images per variant (None = all)
            save_results: Whether to save results to JSON file
            
        Returns:
            Overall statistics
        """
        logger.info(f"Starting drift data sender")
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Endpoint: {self.detect_endpoint} {'(GPU)' if self.use_gpu else '(CPU)'}")
        logger.info(f"Test variants directory: {test_variants_dir}")
        
        # Check API health
        try:
            health_response = requests.get(
                f"{self.api_url}/health",
                timeout=5,
            )
            if health_response.status_code == 200:
                logger.info("✓ API is healthy")
            else:
                logger.warning(f"API health check returned: {health_response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to API: {e}")
            logger.error("Please ensure the serving API is running!")
            return {}
        
        # Find all images
        all_variants = self.find_images(test_variants_dir)
        
        if not all_variants:
            logger.error("No images found in test variants directory")
            return {}
        
        # Filter variants if specified
        if variants:
            all_variants = {
                k: v for k, v in all_variants.items() if k in variants
            }
            if not all_variants:
                logger.error(f"None of the specified variants found: {variants}")
                return {}
        
        # Process each variant
        summaries = []
        start_time = time.time()
        
        for variant_name, images in all_variants.items():
            summary = self.send_batch(
                variant_name,
                images,
                max_images=max_images_per_variant,
            )
            summaries.append(summary)
        
        total_time = time.time() - start_time
        
        # Overall statistics
        overall = {
            "total_variants": len(summaries),
            "total_images": sum(s["total_images"] for s in summaries),
            "total_success": sum(s["success"] for s in summaries),
            "total_errors": sum(s["errors"] for s in summaries),
            "total_detections": sum(s["total_detections"] for s in summaries),
            "total_time_seconds": total_time,
            "variants": summaries,
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total variants processed: {overall['total_variants']}")
        logger.info(f"Total images sent: {overall['total_images']}")
        logger.info(f"Total success: {overall['total_success']}")
        logger.info(f"Total errors: {overall['total_errors']}")
        logger.info(f"Total detections: {overall['total_detections']}")
        logger.info(f"Total time: {overall['total_time_seconds']:.2f} seconds")
        
        # Verify production data storage if enabled
        if self.verify_minio:
            logger.info(f"\n{'='*60}")
            logger.info("PRODUCTION DATA VERIFICATION")
            logger.info(f"{'='*60}")
            verification = self.verify_production_data_storage()
            if verification.get("verified"):
                logger.info(f"✓ Production data bucket: {verification['bucket']}")
                logger.info(f"✓ Date folder: {verification['date']}")
                logger.info(f"✓ Images in bucket: {verification['image_count']}")
                if verification.get("objects"):
                    logger.info(f"✓ Sample objects: {verification['objects'][:3]}")
                overall["minio_verification"] = verification
            else:
                logger.warning(f"✗ MinIO verification failed: {verification.get('reason', verification.get('error'))}")
                overall["minio_verification"] = verification
        
        # Save results if requested
        if save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = Path("drift_sending_results_" + timestamp + ".json")
            
            output_data = {
                "summary": overall,
                "detailed_results": self.results,
            }
            
            with open(results_file, "w") as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"\n✓ Results saved to: {results_file}")
        
        return overall


def check_production_bucket(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str = "production-data",
) -> None:
    """
    Check the status of the production-data bucket.
    
    Args:
        minio_endpoint: MinIO endpoint URL
        minio_access_key: MinIO access key
        minio_secret_key: MinIO secret key
        bucket_name: Name of the bucket to check
    """
    try:
        import boto3
        from botocore.client import Config
        
        client = boto3.client(
            's3',
            endpoint_url=minio_endpoint,
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        logger.info(f"Checking bucket: {bucket_name}")
        
        # Check if bucket exists
        try:
            client.head_bucket(Bucket=bucket_name)
            logger.info(f"✓ Bucket exists: {bucket_name}")
        except:
            logger.error(f"✗ Bucket does not exist: {bucket_name}")
            return
        
        # List all date folders
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Delimiter='/')
        
        date_folders = []
        for page in pages:
            if 'CommonPrefixes' in page:
                date_folders.extend([p['Prefix'].rstrip('/') for p in page['CommonPrefixes']])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Date folders in {bucket_name}:")
        logger.info(f"{'='*60}")
        
        if not date_folders:
            logger.info("No date folders found (bucket is empty)")
            return
        
        # Show statistics for each date folder
        for date_folder in sorted(date_folders):
            pages = paginator.paginate(Bucket=bucket_name, Prefix=f"{date_folder}/")
            objects = []
            for page in pages:
                if 'Contents' in page:
                    objects.extend(page['Contents'])
            
            total_size = sum(obj['Size'] for obj in objects)
            logger.info(f"  {date_folder}: {len(objects)} images, {total_size / 1024 / 1024:.2f} MB")
        
        logger.info(f"\n✓ Total date folders: {len(date_folders)}")
        
    except ImportError:
        logger.error("boto3 not installed. Install with: pip install boto3")
    except Exception as e:
        logger.error(f"Failed to check bucket: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Send test variant images to serving API for drift detection and verify production data storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send test variant images to CPU endpoint
  python scripts/send_drift_data.py --test-dir data/test_variants
  
  # Send to GPU endpoint for faster inference (auto-switches to port 8001)
  python scripts/send_drift_data.py --test-dir data/test_variants --use-gpu
  
  # Send with MinIO verification
  python scripts/send_drift_data.py --test-dir data/test_variants --verify-minio
  
  # Check production-data bucket status
  python scripts/send_drift_data.py --check-bucket
  
  # Send specific variants with delay
  python scripts/send_drift_data.py --variants test_abstract test_brightness --delay 0.5
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--check-bucket",
        action="store_true",
        help="Check production-data bucket status (don't send images)",
    )
    
    # API settings
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test_variants",
        help="Directory containing test variant folders (default: data/test_variants)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the serving API (default: http://localhost:8000 for CPU, http://localhost:8001 for GPU)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU endpoint (/detect-gpu) for faster inference. Requires GPU-enabled API service.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        help="Specific variants to process (e.g., test_abstract test_brightness)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images per variant (default: all)",
    )
    
    # Detection settings
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between requests in seconds (default: 0.0)",
    )
    
    # MinIO settings
    parser.add_argument(
        "--verify-minio",
        action="store_true",
        help="Verify that images are saved to MinIO production-data bucket",
    )
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        default="http://localhost:9000",
        help="MinIO endpoint URL (default: http://localhost:9000)",
    )
    parser.add_argument(
        "--minio-access-key",
        type=str,
        default="minio_admin",
        help="MinIO access key (default: minio_admin)",
    )
    parser.add_argument(
        "--minio-secret-key",
        type=str,
        default="minio_password123",
        help="MinIO secret key (default: minio_password123)",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="production-data",
        help="Production data bucket name (default: production-data)",
    )
    
    # Output settings
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Auto-adjust API URL for GPU if not explicitly set
    if args.use_gpu and args.api_url == "http://localhost:8000":
        args.api_url = "http://localhost:8001"
        logger.info(f"🚀 GPU mode enabled, using GPU API at {args.api_url}")
    
    # Check bucket mode
    if args.check_bucket:
        check_production_bucket(
            minio_endpoint=args.minio_endpoint,
            minio_access_key=args.minio_access_key,
            minio_secret_key=args.minio_secret_key,
            bucket_name=args.bucket_name,
        )
        return
    
    # Send drift data mode
    sender = DriftDataSender(
        api_url=args.api_url,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        delay=args.delay,
        use_gpu=args.use_gpu,
        verify_minio=args.verify_minio,
        minio_endpoint=args.minio_endpoint,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
    )
    
    # Run
    test_variants_dir = Path(args.test_dir)
    sender.run(
        test_variants_dir=test_variants_dir,
        variants=args.variants,
        max_images_per_variant=args.max_images,
        save_results=args.save_results,
    )


if __name__ == "__main__":
    main()
