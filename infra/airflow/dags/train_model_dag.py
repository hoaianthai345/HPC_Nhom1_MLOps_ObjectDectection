"""
Airflow DAG for Model Training with MinIO Data Fetching.

This DAG orchestrates the complete training pipeline:
1. Fetches training data from MinIO bucket
2. Prepares data and configuration
3. Trains YOLO model (yolo11n - nano model for efficiency)
4. Logs model and metrics to MLflow
5. Registers trained model to MLflow Model Registry

Schedule: Manual trigger or cron-based

Note: Uses standard Ultralytics training. Knowledge distillation via custom
teacher/student parameters is not supported by Ultralytics train() API.
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator


default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': ['mlops@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def fetch_data_from_minio(**context):
    """
    Fetch training data from MinIO bucket.
    Downloads images and labels from versioned MinIO storage.
    Supports specifying version or auto-downloading the latest version.
    """
    import boto3
    from botocore.client import Config
    from pathlib import Path
    import os
    import json
    
    print("🗄️  Starting data fetching from MinIO...")
    
    # MinIO configuration
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
    MINIO_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'minio_admin')
    MINIO_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    BUCKET_NAME = os.getenv('MINIO_TRAINING_BUCKET', 'training-data')
    
    # Get version from environment variable or use latest
    DATA_VERSION = os.getenv('DATA_VERSION', None)  # e.g., 'v1.0' or None for latest
    
    # Local paths
    data_dir = Path('/tmp/training_data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MinIO client
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    print(f"📦 Connected to MinIO at {MINIO_ENDPOINT}")
    print(f"📂 Bucket: {BUCKET_NAME}")
    
    # Determine which version to download
    if DATA_VERSION is None:
        print("🔍 No version specified, finding latest version...")
        
        # List all versions (top-level directories)
        try:
            response = s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Delimiter='/'
            )
            
            versions = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    version = prefix['Prefix'].rstrip('/')
                    if version.startswith('v'):  # Only consider version directories
                        versions.append(version)
            
            if not versions:
                print("❌ No versions found in bucket!")
                raise Exception("No data versions available in MinIO")
            
            # Sort versions and get latest (assumes semantic versioning like v1.0, v1.1, v2.0)
            versions.sort(reverse=True)
            DATA_VERSION = versions[0]
            print(f"✅ Using latest version: {DATA_VERSION}")
            
        except Exception as e:
            print(f"❌ Error finding latest version: {e}")
            raise
    else:
        print(f"📌 Using specified version: {DATA_VERSION}")
    
    # Download version info
    try:
        manifest_obj = s3_client.get_object(
            Bucket=BUCKET_NAME,
            Key=f"{DATA_VERSION}/manifest.json"
        )
        manifest = json.loads(manifest_obj['Body'].read().decode('utf-8'))
        print(f"📄 Manifest: {manifest.get('description', 'No description')}")
        if 'stats' in manifest:
            print(f"📊 Dataset stats:")
            for split, stats in manifest['stats'].items():
                print(f"   {split}: {stats.get('images', 0)} images, {stats.get('labels', 0)} labels")
    except Exception as e:
        print(f"⚠️  Warning: Could not read manifest: {e}")
    
    # Create directory structure
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            (data_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Download files from the specified version
    try:
        print(f"\n⬇️  Downloading data from {DATA_VERSION}...")
        
        # List all objects with version prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=BUCKET_NAME,
            Prefix=f"{DATA_VERSION}/"
        )
        
        download_count = 0
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                
                # Skip directories and manifest
                if key.endswith('/') or key.endswith('manifest.json'):
                    continue
                
                # Remove version prefix from key to get local path
                # e.g., "v1.0/train/images/img1.jpg" -> "train/images/img1.jpg"
                relative_key = key[len(DATA_VERSION)+1:]  # +1 for the slash
                
                if not relative_key:  # Skip if empty after stripping
                    continue
                
                # Determine local path (without version prefix)
                local_file = data_dir / relative_key
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                s3_client.download_file(BUCKET_NAME, key, str(local_file))
                download_count += 1
                
                if download_count % 100 == 0:
                    print(f"   Downloaded {download_count} files...")
        
        print(f"✅ Successfully downloaded {download_count} files from {DATA_VERSION}")
        
        # Count files per split
        print("\n📊 Downloaded dataset structure:")
        for split in ['train', 'valid', 'test']:
            split_dir = data_dir / split
            if split_dir.exists():
                images_count = len(list((split_dir / 'images').glob('*')))
                labels_count = len(list((split_dir / 'labels').glob('*')))
                print(f"  {split:6} - Images: {images_count:5} | Labels: {labels_count:5}")
        
        # Push data directory path and version to XCom for next tasks
        context['task_instance'].xcom_push(key='data_dir', value=str(data_dir))
        context['task_instance'].xcom_push(key='data_version', value=DATA_VERSION)
        
        return str(data_dir)
        
    except Exception as e:
        print(f"❌ Error fetching data from MinIO: {e}")
        raise


def prepare_data_yaml(**context):
    """
    Create data.yaml configuration file for YOLO training.
    """
    import yaml
    from pathlib import Path
    
    print("📝 Preparing data.yaml configuration...")
    
    # Get data directory and version from previous task
    data_dir = Path(context['task_instance'].xcom_pull(
        task_ids='fetch_data_from_minio', 
        key='data_dir'
    ))
    data_version = context['task_instance'].xcom_pull(
        task_ids='fetch_data_from_minio',
        key='data_version'
    )
    
    # Class names and count (update based on your dataset)
    class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    nc = len(class_names)
    
    # Create data.yaml
    data_config = {
        'path': str(data_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': nc,
        'names': class_names
    }
    
    yaml_path = data_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml at {yaml_path}")
    print(f"   Data version: {data_version}")
    print(f"   Classes: {class_names}")
    print(f"   Number of classes: {nc}")
    
    # Push yaml path to XCom
    context['task_instance'].xcom_push(key='data_yaml', value=str(yaml_path))
    
    return str(yaml_path)


def download_teacher_model(**context):
    """
    Download or fetch teacher model from MinIO or MLflow Model Registry.
    """
    import boto3
    from botocore.client import Config
    from pathlib import Path
    import os
    
    print("🎓 Fetching teacher model...")
    
    # MinIO configuration
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
    MINIO_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'minio_admin')
    MINIO_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    BUCKET_NAME = 'mlflow'  # Teacher model stored in MLflow bucket
    
    # Teacher model path
    teacher_weights_path = Path('/tmp/teacher_model/yolov11x.pt')
    teacher_weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to download from MinIO first
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        # Check if teacher model exists in model-exports bucket
        try:
            s3_client.download_file(
                'model-exports', 
                'teacher/yolov11x.pt', 
                str(teacher_weights_path)
            )
            print(f"✅ Downloaded teacher model from MinIO")
        except:
            # If not in MinIO, download from Ultralytics pretrained
            print("⬇️  Downloading pretrained YOLOv11x model...")
            from ultralytics import YOLO
            model = YOLO('yolo11x.pt')  # This will auto-download
            
            # Move to teacher path
            import shutil
            shutil.copy('yolo11x.pt', str(teacher_weights_path))
            print(f"✅ Downloaded pretrained teacher model")
        
        # Push teacher model path to XCom
        context['task_instance'].xcom_push(key='teacher_weights', value=str(teacher_weights_path))
        
        return str(teacher_weights_path)
        
    except Exception as e:
        print(f"❌ Error fetching teacher model: {e}")
        raise


def prepare_training_config(**context):
    """
    Prepare training configuration file.
    """
    import yaml
    from pathlib import Path
    
    print("⚙️  Preparing training configuration...")
    
    config_dir = Path('/tmp/training_config')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    training_config = {
        'training': {
            'epochs': int(os.getenv('TRAIN_EPOCHS', '50')),
            'imgsz': 640,
            'batch': int(os.getenv('TRAIN_BATCH_SIZE', '16')),
            'device': os.getenv('TRAIN_DEVICE', '0'),
            'workers': 0,
            'seed': 42,
            'deterministic': True
        },
        'optimization': {
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'warmup_epochs': 3
        },
        'augmentation': {
            'mosaic': 1.0,
            'close_mosaic': 10
        },
        'distillation': {
            'logit_temperature': 3.0,
            'dense_logit_weight': 0.25,
            'sparse_logit_weight': 0.25,
            'box_loss_weight': 0.5,
            'box_objectness_threshold': 0.3
        },
        'logging': {
            'project': 'yolo_training',  # Changed from yolo-distillation
            'name': f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'model_name': 'yolo-nano-model',
            'save': True,
            'save_period': 10,
            'plots': True
        }
    }
    
    config_path = config_dir / 'train_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(training_config, f, default_flow_style=False)
    
    print(f"✅ Created training config at {config_path}")
    
    # Push config path to XCom
    context['task_instance'].xcom_push(key='train_config', value=str(config_path))
    
    return str(config_path)


def train_model(**context):
    """
    Train YOLO model.
    Note: Knowledge distillation is not directly supported via Ultralytics train() parameters.
    Using standard training approach with student model (yolo11n).
    """
    import yaml
    import os
    from pathlib import Path
    from ultralytics import YOLO
    import mlflow
    
    print("🚀 Starting model training...")
    
    # Get paths from previous tasks
    data_yaml = context['task_instance'].xcom_pull(task_ids='prepare_data_yaml', key='data_yaml')
    teacher_weights = context['task_instance'].xcom_pull(task_ids='download_teacher_model', key='teacher_weights')
    train_config = context['task_instance'].xcom_pull(task_ids='prepare_training_config', key='train_config')
    data_version = context['task_instance'].xcom_pull(task_ids='fetch_data_from_minio', key='data_version')
    
    # Load configuration
    with open(train_config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # MLflow configuration
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minio_admin')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment_name = cfg['logging']['project']
    mlflow.set_experiment(experiment_name)
    
    # Enable MLflow in Ultralytics
    from ultralytics import settings as ultra_settings
    ultra_settings.update({"mlflow": True})
    
    # Load student model
    print(f"🎓 Initializing student model: yolo11n.pt")
    student_model = YOLO('yolo11n.pt')
    
    # Note: Teacher model downloaded but not used in standard Ultralytics training
    # Knowledge distillation is not supported via train() parameters in Ultralytics
    print(f"ℹ️  Teacher model available at: {teacher_weights} (for reference)")
    
    run_name = cfg['logging']['name']
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters (including data version for reproducibility)
        mlflow.log_params({
            'data_version': data_version,  # Track which data version was used
            'model_weights': 'yolo11n.pt',
            'data_yaml': data_yaml,
            'epochs': cfg['training']['epochs'],
            'imgsz': cfg['training']['imgsz'],
            'batch': cfg['training']['batch'],
            'optimizer': cfg['optimization']['optimizer'],
            'lr0': cfg['optimization']['lr0'],
            'seed': cfg['training']['seed'],
        })
        # Note: Not logging distillation config as it's not used in training
        
        print(f"🏋️  Training student model...")
        print(f"   Epochs: {cfg['training']['epochs']}")
        print(f"   Batch size: {cfg['training']['batch']}")
        print(f"   Device: {cfg['training']['device']}")
        print(f"   Note: Using standard training (distillation not supported in Ultralytics CLI)")
        
        # Train student model (standard training without distillation)
        results = student_model.train(
            data=data_yaml,
            epochs=cfg['training']['epochs'],
            imgsz=cfg['training']['imgsz'],
            batch=cfg['training']['batch'],
            device=cfg['training']['device'],
            workers=cfg['training']['workers'],
            seed=cfg['training']['seed'],
            deterministic=cfg['training']['deterministic'],
            optimizer=cfg['optimization']['optimizer'],
            lr0=cfg['optimization']['lr0'],
            warmup_epochs=cfg['optimization']['warmup_epochs'],
            mosaic=cfg['augmentation']['mosaic'],
            close_mosaic=cfg['augmentation']['close_mosaic'],
            project=cfg['logging']['project'],
            name=cfg['logging']['name'],
            save=cfg['logging']['save'],
            save_period=cfg['logging']['save_period'],
            plots=cfg['logging']['plots'],
        )
        
        print("✅ Training completed!")
        
        # Log artifacts
        save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
        if save_dir and save_dir.exists():
            best_pt = save_dir / 'weights' / 'best.pt'
            last_pt = save_dir / 'weights' / 'last.pt'
            
            if best_pt.exists():
                print(f"📦 Logging best model weights...")
                mlflow.log_artifact(str(best_pt), artifact_path='weights')
            
            if last_pt.exists():
                mlflow.log_artifact(str(last_pt), artifact_path='weights')
            
            # Log training plots and results
            for ext in ('*.png', '*.csv'):
                for f in save_dir.glob(ext):
                    mlflow.log_artifact(str(f), artifact_path='results')
        
        # Register model to MLflow Model Registry
        if save_dir and (save_dir / 'weights' / 'best.pt').exists():
            print("\n📦 Registering model to MLflow Model Registry...")
            
            model_name = cfg['logging']['model_name']
            model_uri = f"runs:/{run.info.run_id}/weights/best.pt"
            
            try:
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                    tags={
                        "training_date": datetime.now().isoformat(),
                        "framework": "ultralytics",
                        "model_type": "yolo11n",
                        "training_method": "standard"
                    }
                )
                
                print(f"✅ Model registered: {model_name} version {model_version.version}")
                
                # Transition to Staging
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
                
                print(f"🎯 Model transitioned to Staging stage")
                
                # Push model info to XCom
                context['task_instance'].xcom_push(key='model_name', value=model_name)
                context['task_instance'].xcom_push(key='model_version', value=model_version.version)
                context['task_instance'].xcom_push(key='run_id', value=run.info.run_id)
                
            except Exception as e:
                print(f"⚠️  Warning: Could not register model: {e}")
        
        return run.info.run_id


def upload_model_to_minio(**context):
    """
    Upload trained model to MinIO for backup and versioning.
    """
    import boto3
    from botocore.client import Config
    from pathlib import Path
    import os
    
    print("☁️  Uploading trained model to MinIO...")
    
    # Get model info from previous task
    model_name = context['task_instance'].xcom_pull(task_ids='train_model', key='model_name')
    model_version = context['task_instance'].xcom_pull(task_ids='train_model', key='model_version')
    run_id = context['task_instance'].xcom_pull(task_ids='train_model', key='run_id')
    
    if not all([model_name, model_version, run_id]):
        print("⚠️  Missing model information, skipping MinIO upload")
        return
    
    # MinIO configuration
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
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
    
    # Find the trained model file (from MLflow artifacts)
    # Typically stored in /tmp or working directory
    model_path = Path(f"yolo-distillation/training_*").glob('weights/best.pt')
    model_files = list(model_path)
    
    if not model_files:
        print("⚠️  Could not find trained model file")
        return
    
    model_file = model_files[0]
    
    # Upload to MinIO
    s3_key = f"trained_models/{model_name}/v{model_version}/best.pt"
    
    try:
        s3_client.upload_file(
            str(model_file),
            BUCKET_NAME,
            s3_key
        )
        print(f"✅ Uploaded model to MinIO: s3://{BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"⚠️  Warning: Could not upload to MinIO: {e}")


def send_training_notification(**context):
    """
    Send notification about training completion.
    """
    model_name = context['task_instance'].xcom_pull(task_ids='train_model', key='model_name')
    model_version = context['task_instance'].xcom_pull(task_ids='train_model', key='model_version')
    run_id = context['task_instance'].xcom_pull(task_ids='train_model', key='run_id')
    
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📦 Model: {model_name}")
    print(f"🔢 Version: {model_version}")
    print(f"🆔 MLflow Run ID: {run_id}")
    print(f"🎯 Stage: Staging")
    print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Here you can add email notification, Slack webhook, etc.
    return {
        'status': 'success',
        'model_name': model_name,
        'model_version': model_version,
        'run_id': run_id
    }


# Define the DAG
with DAG(
    'train_model',
    default_args=default_args,
    description='Train YOLO model with data from MinIO',
    schedule=None,  # Manual trigger, or use cron: '@daily', '@weekly'
    start_date=datetime(2026, 2, 1),  # Fixed date for manually triggered DAG
    catchup=False,
    tags=['training', 'mlflow', 'yolo'],
) as dag:
    
    # Task 1: Fetch training data from MinIO
    fetch_data = PythonOperator(
        task_id='fetch_data_from_minio',
        python_callable=fetch_data_from_minio,
    )
    
    # Task 2: Prepare data.yaml configuration
    prepare_yaml = PythonOperator(
        task_id='prepare_data_yaml',
        python_callable=prepare_data_yaml,
    )
    
    # Task 3: Download teacher model
    fetch_teacher = PythonOperator(
        task_id='download_teacher_model',
        python_callable=download_teacher_model,
    )
    
    # Task 4: Prepare training configuration
    prepare_config = PythonOperator(
        task_id='prepare_training_config',
        python_callable=prepare_training_config,
    )
    
    # Task 5: Train the model
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        execution_timeout=timedelta(hours=6),  # Allow up to 6 hours for training
    )
    
    # Task 6: Upload model to MinIO
    upload_to_minio = PythonOperator(
        task_id='upload_model_to_minio',
        python_callable=upload_model_to_minio,
    )
    
    # Task 7: Send notification
    notify = PythonOperator(
        task_id='send_training_notification',
        python_callable=send_training_notification,
    )
    
    # Define task dependencies
    fetch_data >> prepare_yaml
    fetch_teacher >> train
    prepare_yaml >> train
    prepare_config >> train
    train >> upload_to_minio >> notify
