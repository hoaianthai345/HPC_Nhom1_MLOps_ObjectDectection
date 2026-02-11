#!/usr/bin/env python3
"""
Train Teacher Model (YOLOv26x) for Knowledge Distillation.

This script trains a larger, more accurate YOLO model (yolo26l) that can be used as:
1. A teacher model for knowledge distillation
2. A baseline for comparison
3. A production model if performance is sufficient

The trained model is:
- Logged to MLflow with 'teacher' tag
- Registered to MLflow Model Registry
- Optionally uploaded to MinIO for easy retrieval

Usage:
    python scripts/train_teacher_model.py --data data/processed/data.yaml --epochs 100
    
    # With custom parameters
    python scripts/train_teacher_model.py \
        --data data/raw/data.yaml \
        --epochs 100 \
        --batch 32 \
        --imgsz 640 \
        --device 0 \
        --name teacher_yolo11l
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import yaml

import mlflow
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLO teacher model (yolo26l)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument(
        '--data', 
        type=str,
        default='data/raw/data.yaml',
        help='Path to data.yaml configuration file'
    )
    
    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='yolo26x.pt',
        help='Model architecture (yolo26x.pt recommended for teacher)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (0 for GPU, cpu for CPU)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )
    
    # Optimization
    parser.add_argument(
        '--optimizer',
        type=str,
        default='AdamW',
        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
        help='Optimizer algorithm'
    )
    
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=3,
        help='Number of warmup epochs'
    )
    
    # Augmentation
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='Mosaic augmentation probability'
    )
    
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        help='MixUp augmentation probability'
    )
    
    # Logging
    parser.add_argument(
        '--project',
        type=str,
        default='teacher_training',
        help='Project name for saving runs'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Run name (default: teacher_YYYYMMDD_HHMMSS)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='yolo-teacher-model',
        help='Model name for MLflow Model Registry'
    )
    
    # MLflow
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: from env or http://localhost:5000)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='teacher_training',
        help='MLflow experiment name'
    )
    
    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (epochs)'
    )
    
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow logging'
    )
    
    parser.add_argument(
        '--upload-minio',
        action='store_true',
        help='Upload trained model to MinIO'
    )
    
    return parser.parse_args()


def setup_mlflow(args):
    """Configure MLflow tracking."""
    if args.no_mlflow:
        print("⚠️  MLflow logging disabled")
        return None
    
    # Set MLflow tracking URI
    mlflow_uri = args.mlflow_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Configure S3 backend for MinIO
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minio_admin')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    # Set experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"✅ MLflow configured:")
    print(f"   Tracking URI: {mlflow_uri}")
    print(f"   Experiment: {args.experiment_name}")
    
    return mlflow_uri


def upload_to_minio(model_path, model_name, model_version):
    """Upload trained model to MinIO for backup."""
    try:
        import boto3
        from botocore.client import Config
        
        print("\n☁️  Uploading model to MinIO...")
        
        # MinIO configuration
        MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
        MINIO_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'minio_admin')
        MINIO_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio_password123')
        BUCKET_NAME = 'model-exports'
        
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        # Upload to teacher folder
        s3_key = f"teacher/{model_name}/v{model_version}/best.pt"
        
        s3_client.upload_file(
            str(model_path),
            BUCKET_NAME,
            s3_key
        )
        
        print(f"✅ Uploaded to MinIO: s3://{BUCKET_NAME}/{s3_key}")
        return f"s3://{BUCKET_NAME}/{s3_key}"
        
    except Exception as e:
        print(f"⚠️  Warning: Could not upload to MinIO: {e}")
        return None


def main():
    """Main training function."""
    args = parse_args()
    
    print("="*60)
    print("🎓 TEACHER MODEL TRAINING")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Validate data file
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    # Generate run name
    run_name = args.name or f"teacher_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup MLflow
    mlflow_uri = setup_mlflow(args)
    
    # Enable MLflow in Ultralytics if not disabled
    if not args.no_mlflow:
        from ultralytics import settings as ultra_settings
        ultra_settings.update({"mlflow": True})
    
    # Load model
    print(f"\n🎓 Loading teacher model: {args.model}")
    model = YOLO(args.model)
    
    # Start MLflow run
    if not args.no_mlflow:
        run = mlflow.start_run(run_name=run_name)
        
        # Log parameters
        mlflow.log_params({
            'model': args.model,
            'data': str(data_path),
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'device': args.device,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'warmup_epochs': args.warmup_epochs,
            'seed': args.seed,
            'model_type': 'teacher',
            'role': 'teacher_model'
        })
        
        print(f"📊 MLflow Run ID: {run.info.run_id}")
    
    try:
        # Train model
        print("\n🏋️  Starting training...")
        print(f"   Project: {args.project}")
        print(f"   Name: {run_name}")
        
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            optimizer=args.optimizer,
            lr0=args.lr0,
            warmup_epochs=args.warmup_epochs,
            mosaic=args.mosaic,
            mixup=args.mixup,
            seed=args.seed,
            deterministic=True,
            project=args.project,
            name=run_name,
            save=True,
            save_period=args.save_period,
            patience=args.patience,
            plots=True,
            verbose=True
        )
        
        print("\n✅ Training completed!")
        
        # Get save directory
        save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
        
        if save_dir and save_dir.exists():
            best_model = save_dir / 'weights' / 'best.pt'
            last_model = save_dir / 'weights' / 'last.pt'
            
            print(f"\n📦 Model saved:")
            print(f"   Best: {best_model}")
            print(f"   Last: {last_model}")
            
            # Register to MLflow Model Registry
            if not args.no_mlflow and best_model.exists():
                print("\n📦 Registering model to MLflow Model Registry...")
                
                model_uri = f"runs:/{run.info.run_id}/weights/best.pt"
                
                try:
                    model_version = mlflow.register_model(
                        model_uri=model_uri,
                        name=args.model_name,
                        tags={
                            "training_date": datetime.now().isoformat(),
                            "framework": "ultralytics",
                            "model_type": "yolo11l",
                            "role": "teacher",
                            "training_method": "standard"
                        }
                    )
                    
                    print(f"✅ Model registered: {args.model_name} version {model_version.version}")
                    
                    # Get validation metrics
                    if hasattr(results, 'results_dict'):
                        metrics = results.results_dict
                        print(f"\n📈 Final Metrics:")
                        print(f"   mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
                        print(f"   mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
                    
                    # Upload to MinIO if requested
                    if args.upload_minio:
                        minio_path = upload_to_minio(
                            best_model, 
                            args.model_name, 
                            model_version.version
                        )
                        if minio_path and not args.no_mlflow:
                            mlflow.log_param('minio_path', minio_path)
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not register model: {e}")
            
    finally:
        if not args.no_mlflow:
            mlflow.end_run()
    
    print("\n" + "="*60)
    print("🎉 TEACHER MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"📂 Results saved to: {save_dir if save_dir else 'N/A'}")
    if not args.no_mlflow:
        print(f"📊 MLflow URI: {mlflow_uri}")
        print(f"🔍 View in MLflow: {mlflow_uri}/#/experiments")
    print("="*60)


if __name__ == '__main__':
    main()
