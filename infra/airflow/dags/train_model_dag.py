"""
Airflow DAG for Model Training with MinIO Data Fetching and Auto-Promotion.

This DAG orchestrates the complete training pipeline:
1. Fetches training data from MinIO bucket
2. Downloads finetuned teacher model from MLflow Model Registry
3. Prepares data and configuration for training
4. Trains YOLO student model using Knowledge Distillation
5. Logs student model and metrics to MLflow
6. Registers trained student model to MLflow Model Registry (with 'staging' alias)
7. Evaluates model performance on validation set
8. Automatically promotes to Production if metrics pass thresholds (sets 'production' alias)
9. Sends notification with training and promotion results

Configuration (via Airflow Variables or Environment Variables):
Priority: Airflow Variable > Environment Variable > Default

MinIO/Storage:
- MINIO_ENDPOINT (default: 'http://minio:9000')
- AWS_ACCESS_KEY_ID (default: 'minio_admin')
- AWS_SECRET_ACCESS_KEY (default: 'minio_password123')
- MINIO_TRAINING_BUCKET (default: 'training-data')
- DATA_VERSION (default: None - uses latest)

MLflow:
- MLFLOW_TRACKING_URI (default: 'http://mlflow_server:5000')

Teacher Model:
- TEACHER_MODEL_NAME (default: 'yolo-teacher-model')
- TEACHER_MODEL_ALIAS (default: 'production')

Training:
- TRAIN_EPOCHS (default: 1)
- TRAIN_BATCH_SIZE (default: 16)
- TRAIN_DEVICE (default: '0')

Promotion Thresholds:
- PROMOTION_MAP50_THRESHOLD (default: 0.5)
- PROMOTION_MAP50_95_THRESHOLD (default: 0.3)
- PROMOTION_PRECISION_THRESHOLD (default: 0.4)
- PROMOTION_RECALL_THRESHOLD (default: 0.4)

To set Airflow Variables via UI:
Admin > Variables > Add
Or via CLI:
airflow variables set TRAIN_EPOCHS 10
airflow variables set PROMOTION_MAP50_THRESHOLD 0.6

Schedule: Manual trigger or cron-based

Prerequisites:
- Teacher model must be trained and registered in MLflow beforehand
  Run: python scripts/train_teacher_model.py --data <data.yaml> --epochs <N>
- Training data must be available in MinIO bucket

Note: Uses Knowledge Distillation training via custom training script
(training_pipeline/src/train.py) with teacher/student model configuration.
Uses modern MLflow alias-based model registry (not deprecated stages).
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.sdk import Variable
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator


def get_config(key: str, default=None):
    """
    Get configuration value from Airflow Variable with fallback to environment variable.
    Priority: Airflow Variable > Environment Variable > Default value
    
    Args:
        key: Configuration key name
        default: Default value if not found in Variable or environment
        
    Returns:
        Configuration value
    """
    try:
        # Try to get from Airflow Variable first (raises exception if not found)
        return Variable.get(key)
    except Exception:
        # Fall back to environment variable if Variable not found or any error occurs
        return os.getenv(key, default)


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
    
    # MinIO configuration (from Airflow Variables or environment)
    MINIO_ENDPOINT = get_config('MINIO_ENDPOINT', 'http://minio:9000')
    MINIO_ACCESS_KEY = get_config('AWS_ACCESS_KEY_ID', 'minio_admin')
    MINIO_SECRET_KEY = get_config('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    BUCKET_NAME = get_config('MINIO_TRAINING_BUCKET', 'training-data')
    
    # Get version from Airflow Variable or use latest
    DATA_VERSION = get_config('DATA_VERSION', None)  # e.g., 'v1.0' or None for latest
    
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
    Download teacher model from MLflow Model Registry.
    The teacher model should be finetuned and logged to MLflow beforehand.
    """
    import mlflow
    from pathlib import Path
    import os
    import shutil
    
    print("🎓 Fetching teacher model from MLflow...")
    
    # MLflow configuration (from Airflow Variables or environment)
    MLFLOW_TRACKING_URI = get_config('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Configure AWS/MinIO credentials for MLflow artifact storage
    os.environ['AWS_ACCESS_KEY_ID'] = get_config('AWS_ACCESS_KEY_ID', 'minio_admin')
    os.environ['AWS_SECRET_ACCESS_KEY'] = get_config('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = get_config('MINIO_ENDPOINT', 'http://minio:9000')
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    # Teacher model configuration from Airflow Variables or environment
    teacher_model_name = get_config('TEACHER_MODEL_NAME', 'yolo-teacher-model')
    teacher_model_alias = get_config('TEACHER_MODEL_ALIAS', 'production')  # or version number
    
    # Local path for teacher model
    teacher_weights_path = Path('/tmp/teacher_model/best.pt')
    teacher_weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 MLflow Configuration:")
    print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"   Model Name: {teacher_model_name}")
    print(f"   Alias: {teacher_model_alias}")
    
    try:
        # Use MlflowClient to get model version by alias (modern approach)
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        print(f"📥 Fetching model version with alias '{teacher_model_alias}'...")
        
        # Get model version by alias
        model_version = client.get_model_version_by_alias(teacher_model_name, teacher_model_alias)
        version_number = model_version.version
        run_id = model_version.run_id
        
        print(f"✓ Found model version: {version_number}")
        print(f"✓ Run ID: {run_id}")
        
        # Download directly from run artifacts (YOLO logs to weights/ path)
        # Try common paths where YOLO models might be stored
        artifact_paths = ['weights/best.pt', 'model/best.pt', 'best.pt']
        downloaded_path = None
        
        for artifact_path in artifact_paths:
            try:
                teacher_model_uri = f"runs:/{run_id}/{artifact_path}"
                print(f"📥 Trying to download from: {teacher_model_uri}")
                
                downloaded_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=teacher_model_uri,
                    dst_path=str(teacher_weights_path.parent)
                )
                print(f"✅ Successfully downloaded from {artifact_path}")
                break
            except Exception as e:
                print(f"   ✗ Not found at {artifact_path}")
                continue
        
        if not downloaded_path:
            raise FileNotFoundError(
                f"Could not find teacher model in any of the expected paths: {artifact_paths}"
            )
        
        print(f"✅ Downloaded teacher model from MLflow")
        print(f"   Run ID: {run_id}")
        print(f"   Downloaded to: {downloaded_path}")
        
        # Handle the downloaded file
        downloaded_path_obj = Path(downloaded_path)
        
        # If it's already the .pt file at target location, we're done
        if downloaded_path_obj.resolve() == teacher_weights_path.resolve():
            print(f"✅ Model already at target location: {teacher_weights_path}")
        elif downloaded_path_obj.is_file() and downloaded_path_obj.suffix == '.pt':
            # It's a .pt file but not at target, copy it
            shutil.copy(str(downloaded_path_obj), str(teacher_weights_path))
            print(f"📁 Copied model file to: {teacher_weights_path}")
        else:
            # If it's a directory, look for .pt files
            pt_files = list(downloaded_path_obj.rglob('*.pt'))
            
            if pt_files:
                # Use the first .pt file found (or best.pt if it exists)
                best_pt = None
                for pt_file in pt_files:
                    if pt_file.name == 'best.pt':
                        best_pt = pt_file
                        break
                
                model_file = best_pt if best_pt else pt_files[0]
                
                # Only copy if not already at target
                if model_file.resolve() != teacher_weights_path.resolve():
                    shutil.copy(str(model_file), str(teacher_weights_path))
                    print(f"📁 Copied model file to: {teacher_weights_path}")
                else:
                    print(f"✅ Model already at target location: {teacher_weights_path}")
            else:
                raise FileNotFoundError(f"No .pt model file found in downloaded artifacts at {downloaded_path}")
        
        if not teacher_weights_path.exists():
            raise FileNotFoundError(f"Teacher model file not found at {teacher_weights_path}")
        
        print(f"✅ Teacher model ready at: {teacher_weights_path}")
        
    except Exception as e:
        print(f"❌ Error downloading teacher model from MLflow: {e}")
        print(f"\n⚠️  Make sure:")
        print(f"   1. Teacher model is trained and registered in MLflow")
        print(f"   2. Model name '{teacher_model_name}' exists in MLflow Model Registry")
        print(f"   3. Model has alias '{teacher_model_alias}' set (not stage)")
        print(f"   4. Model artifacts exist in MLflow run (check MinIO bucket)")
        print(f"\nYou can train and register teacher model using:")
        print(f"   python scripts/train_teacher_model.py --data <data.yaml> --epochs <N>")
        raise
    
    # Push teacher model path to XCom
    context['task_instance'].xcom_push(key='teacher_weights', value=str(teacher_weights_path))
    
    return str(teacher_weights_path)


def prepare_training_config(**context):
    """
    Prepare training configuration file.
    """
    import yaml
    from pathlib import Path
    
    print("⚙️  Preparing training configuration...")
    
    config_dir = Path('/tmp/training_config')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Training configuration for Knowledge Distillation (from Airflow Variables or environment)
    training_config = {
        'training': {
            'epochs': int(get_config('TRAIN_EPOCHS', '1')),
            'imgsz': 640,
            'batch': int(get_config('TRAIN_BATCH_SIZE', '16')),
            'device': get_config('TRAIN_DEVICE', '0'),
            'workers': 2,  # Reduced to avoid shared memory issues in Docker
            'seed': 42,
            'deterministic': True
        },
        'optimization': {
            'optimizer': 'auto',
            'lr0': 0.01,
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
            'project': 'yolo_training',
            'name': f'kd_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
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
    print(f"   Distillation: Enabled")
    print(f"   Teacher-Student training: Active")
    
    # Push config path to XCom
    context['task_instance'].xcom_push(key='train_config', value=str(config_path))
    
    return str(config_path)


def train_model(**context):
    """
    Train YOLO model using Knowledge Distillation.
    Calls the custom training script from training_pipeline/src/train.py
    which implements teacher-student distillation.
    """
    import yaml
    import os
    import subprocess
    from pathlib import Path
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("🚀 Starting Knowledge Distillation training...")
    
    # Get paths from previous tasks
    data_yaml = context['task_instance'].xcom_pull(task_ids='prepare_data_yaml', key='data_yaml')
    teacher_weights = context['task_instance'].xcom_pull(task_ids='download_teacher_model', key='teacher_weights')
    train_config = context['task_instance'].xcom_pull(task_ids='prepare_training_config', key='train_config')
    data_version = context['task_instance'].xcom_pull(task_ids='fetch_data_from_minio', key='data_version')
    
    # Load configuration
    with open(train_config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # MLflow configuration (from Airflow Variables or environment)
    MLFLOW_TRACKING_URI = get_config('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
    
    # Prepare environment variables for training script
    train_env = os.environ.copy()
    train_env['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    train_env['AWS_ACCESS_KEY_ID'] = get_config('AWS_ACCESS_KEY_ID', 'minio_admin')
    train_env['AWS_SECRET_ACCESS_KEY'] = get_config('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    train_env['MLFLOW_S3_ENDPOINT_URL'] = get_config('MINIO_ENDPOINT', 'http://minio:9000')
    train_env['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    # Path to training script
    training_script = '/opt/airflow/training_pipeline/src/train.py'
    student_weights = '/opt/airflow/training_pipeline/src/yolo26n.pt'
    
    # Build command to run training script (use -u for unbuffered output)
    cmd = [
        'python',
        '-u',  # Unbuffered output for real-time logging
        training_script,
        train_config,
        '--teacher-weights', teacher_weights,
        '--student-weights', student_weights,
        '--data', data_yaml,
        '--mlflow-tracking-uri', MLFLOW_TRACKING_URI,
        '--mlflow-experiment', cfg['logging']['project'],
        '--mlflow-run-name', cfg['logging']['name'],
    ]
    
    print(f"🎓 Training Configuration:")
    print(f"   Teacher model: {teacher_weights}")
    print(f"   Student model: {student_weights}")
    print(f"   Data: {data_yaml}")
    print(f"   Config: {train_config}")
    print(f"   Epochs: {cfg['training']['epochs']}")
    print(f"   Batch size: {cfg['training']['batch']}")
    print(f"   Device: {cfg['training']['device']}")
    print(f"   Data version: {data_version}")
    print(f"   Method: Knowledge Distillation")
    print(f"\n📜 Running command:")
    print(f"   {' '.join(cmd)}")
    
    # Execute training script with real-time output streaming
    print("\n📝 Training Output (streaming):")
    print("-" * 80)
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            env=train_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end='', flush=True)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        
        print("-" * 80)
        print("\n✅ Knowledge Distillation training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error during training: {e}")
        raise
    
    # Connect to MLflow to get the run information
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Find the most recent run in the experiment
    experiment = mlflow.get_experiment_by_name(cfg['logging']['project'])
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs:
            run = runs[0]
            run_id = run.info.run_id
            
            print(f"\n📊 MLflow Run Information:")
            print(f"   Run ID: {run_id}")
            print(f"   Experiment: {cfg['logging']['project']}")
            
            # Get model from MLflow Model Registry
            model_name = cfg['logging']['model_name']
            
            # Get the latest version of the model
            try:
                model_versions = client.search_model_versions(f"name='{model_name}'")
                if model_versions:
                    # Sort by version number descending
                    latest_version = max(model_versions, key=lambda v: int(v.version))
                    model_version = latest_version.version
                    
                    print(f"\n📦 Model Registry Information:")
                    print(f"   Model: {model_name}")
                    print(f"   Version: {model_version}")
                    print(f"   Stage: {latest_version.current_stage}")
                    
                    # Push model info to XCom
                    context['task_instance'].xcom_push(key='model_name', value=model_name)
                    context['task_instance'].xcom_push(key='model_version', value=model_version)
                    context['task_instance'].xcom_push(key='run_id', value=run_id)
                else:
                    print(f"⚠️  No model versions found for {model_name}")
                    raise Exception(f"Model {model_name} not found in registry")
                    
            except Exception as e:
                print(f"⚠️  Error retrieving model information: {e}")
                raise
            
            return run_id
        else:
            print("⚠️  No runs found in experiment")
            raise Exception("Training run not found in MLflow")
    else:
        print(f"⚠️  Experiment {cfg['logging']['project']} not found")
        raise Exception(f"MLflow experiment not found")


def evaluate_model_performance(**context):
    """
    Evaluate trained model performance on validation set.
    """
    import yaml
    import os
    from pathlib import Path
    from ultralytics import YOLO
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("📊 Evaluating model performance...")
    
    # Get model info from previous task
    model_name = context['task_instance'].xcom_pull(task_ids='train_model', key='model_name')
    model_version = context['task_instance'].xcom_pull(task_ids='train_model', key='model_version')
    run_id = context['task_instance'].xcom_pull(task_ids='train_model', key='run_id')
    data_yaml = context['task_instance'].xcom_pull(task_ids='prepare_data_yaml', key='data_yaml')
    
    # MLflow configuration (from Airflow Variables or environment)
    MLFLOW_TRACKING_URI = get_config('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['AWS_ACCESS_KEY_ID'] = get_config('AWS_ACCESS_KEY_ID', 'minio_admin')
    os.environ['AWS_SECRET_ACCESS_KEY'] = get_config('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = get_config('MINIO_ENDPOINT', 'http://minio:9000')
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Download model from MLflow run artifacts (more reliable than registered model)
    model_uri = f"runs:/{run_id}/weights/best.pt"
    
    # Use mlflow.artifacts API (MLflow 2.x)
    import mlflow.artifacts
    model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    
    print(f"📥 Downloaded model from: {model_uri}")
    print(f"📂 Model saved to: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    print("🔍 Running validation...")
    val_results = model.val(data=data_yaml, split='val', workers=0)
    
    # Extract metrics
    metrics = {
        'val_mAP50': float(val_results.box.map50),  # mAP@0.5
        'val_mAP50_95': float(val_results.box.map),  # mAP@0.5:0.95
        'val_precision': float(val_results.box.mp),  # mean precision
        'val_recall': float(val_results.box.mr),  # mean recall
    }
    
    print("\n📈 Validation Metrics:")
    print(f"   mAP@0.5:    {metrics['val_mAP50']:.4f}")
    print(f"   mAP@0.5:0.95: {metrics['val_mAP50_95']:.4f}")
    print(f"   Precision:   {metrics['val_precision']:.4f}")
    print(f"   Recall:      {metrics['val_recall']:.4f}")
    
    # Update MLflow run with validation metrics
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
    
    # Update model version with metrics in description
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=f"Validation metrics - mAP@0.5: {metrics['val_mAP50']:.4f}, mAP@0.5:0.95: {metrics['val_mAP50_95']:.4f}, Precision: {metrics['val_precision']:.4f}, Recall: {metrics['val_recall']:.4f}"
    )
    
    # Push metrics to XCom for promotion decision
    context['task_instance'].xcom_push(key='val_metrics', value=metrics)
    
    return metrics


def promote_to_production(**context):
    """
    Promote model to Production if it passes performance thresholds.
    """
    import os
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("🎯 Checking model promotion criteria...")
    
    # Get model info and metrics
    model_name = context['task_instance'].xcom_pull(task_ids='train_model', key='model_name')
    model_version = context['task_instance'].xcom_pull(task_ids='train_model', key='model_version')
    metrics = context['task_instance'].xcom_pull(task_ids='evaluate_model_performance', key='val_metrics')
    
    # MLflow configuration (from Airflow Variables or environment)
    MLFLOW_TRACKING_URI = get_config('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Performance thresholds (from Airflow Variables or environment)
    threshold_map50 = float(get_config('PROMOTION_MAP50_THRESHOLD', '0.5'))  # Default: 50% mAP@0.5
    threshold_map50_95 = float(get_config('PROMOTION_MAP50_95_THRESHOLD', '0.3'))  # Default: 30% mAP@0.5:0.95
    threshold_precision = float(get_config('PROMOTION_PRECISION_THRESHOLD', '0.4'))  # Default: 40% precision
    threshold_recall = float(get_config('PROMOTION_RECALL_THRESHOLD', '0.4'))  # Default: 40% recall
    
    print("\n📋 Performance Thresholds:")
    print(f"   mAP@0.5:      >= {threshold_map50:.2f}")
    print(f"   mAP@0.5:0.95:  >= {threshold_map50_95:.2f}")
    print(f"   Precision:     >= {threshold_precision:.2f}")
    print(f"   Recall:        >= {threshold_recall:.2f}")
    
    print("\n📊 Current Model Performance:")
    print(f"   mAP@0.5:      {metrics['val_mAP50']:.4f}")
    print(f"   mAP@0.5:0.95:  {metrics['val_mAP50_95']:.4f}")
    print(f"   Precision:     {metrics['val_precision']:.4f}")
    print(f"   Recall:        {metrics['val_recall']:.4f}")
    
    # Check if model passes all thresholds
    passes_map50 = metrics['val_mAP50'] >= threshold_map50
    passes_map50_95 = metrics['val_mAP50_95'] >= threshold_map50_95
    passes_precision = metrics['val_precision'] >= threshold_precision
    passes_recall = metrics['val_recall'] >= threshold_recall
    
    passes_all = passes_map50 and passes_map50_95 and passes_precision and passes_recall
    
    print("\n✅ Threshold Check Results:")
    print(f"   mAP@0.5:      {'✓ PASS' if passes_map50 else '✗ FAIL'}")
    print(f"   mAP@0.5:0.95:  {'✓ PASS' if passes_map50_95 else '✗ FAIL'}")
    print(f"   Precision:     {'✓ PASS' if passes_precision else '✗ FAIL'}")
    print(f"   Recall:        {'✓ PASS' if passes_recall else '✗ FAIL'}")
    
    if passes_all:
        print("\n🚀 Model passes all thresholds! Promoting to Production...")
        
        # Set alias to Production (modern approach)
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=model_version
        )
        
        print(f"✅ Model {model_name} version {model_version} promoted to Production!")
        print(f"   Alias: production")
        
        # Push promotion status to XCom
        context['task_instance'].xcom_push(key='promoted', value=True)
        context['task_instance'].xcom_push(key='promotion_reason', value='Passed all performance thresholds')
        
        return {
            'promoted': True,
            'reason': 'Passed all performance thresholds',
            'metrics': metrics
        }
    else:
        print("\n⚠️  Model does not meet production criteria. Keeping in Staging.")
        
        failed_checks = []
        if not passes_map50:
            failed_checks.append(f"mAP@0.5: {metrics['val_mAP50']:.4f} < {threshold_map50:.2f}")
        if not passes_map50_95:
            failed_checks.append(f"mAP@0.5:0.95: {metrics['val_mAP50_95']:.4f} < {threshold_map50_95:.2f}")
        if not passes_precision:
            failed_checks.append(f"Precision: {metrics['val_precision']:.4f} < {threshold_precision:.2f}")
        if not passes_recall:
            failed_checks.append(f"Recall: {metrics['val_recall']:.4f} < {threshold_recall:.2f}")
        
        failure_reason = '; '.join(failed_checks)
        print(f"   Failed checks: {failure_reason}")
        
        # Push promotion status to XCom
        context['task_instance'].xcom_push(key='promoted', value=False)
        context['task_instance'].xcom_push(key='promotion_reason', value=failure_reason)
        
        return {
            'promoted': False,
            'reason': failure_reason,
            'metrics': metrics
        }


def send_training_notification(**context):
    """
    Send notification about training completion and promotion status.
    """
    model_name = context['task_instance'].xcom_pull(task_ids='train_model', key='model_name')
    model_version = context['task_instance'].xcom_pull(task_ids='train_model', key='model_version')
    run_id = context['task_instance'].xcom_pull(task_ids='train_model', key='run_id')
    promoted = context['task_instance'].xcom_pull(task_ids='promote_to_production', key='promoted')
    promotion_reason = context['task_instance'].xcom_pull(task_ids='promote_to_production', key='promotion_reason')
    metrics = context['task_instance'].xcom_pull(task_ids='evaluate_model_performance', key='val_metrics')
    
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print(f"📦 Model: {model_name}")
    print(f"🔢 Version: {model_version}")
    print(f"🆔 MLflow Run ID: {run_id}")
    
    if metrics:
        print("\n📊 Performance Metrics:")
        print(f"   mAP@0.5:      {metrics.get('val_mAP50', 0):.4f}")
        print(f"   mAP@0.5:0.95:  {metrics.get('val_mAP50_95', 0):.4f}")
        print(f"   Precision:     {metrics.get('val_precision', 0):.4f}")
        print(f"   Recall:        {metrics.get('val_recall', 0):.4f}")
    
    if promoted:
        print("\n🚀 Status: PROMOTED TO PRODUCTION")
        print(f"   Stage: Production")
        print(f"   Alias: production")
        print(f"   Reason: {promotion_reason}")
    else:
        print("\n⚠️  Status: KEPT IN STAGING")
        print(f"   Stage: Staging")
        print(f"   Reason: {promotion_reason}")
    
    print(f"\n🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Here you can add email notification, Slack webhook, etc.
    return {
        'status': 'success',
        'model_name': model_name,
        'model_version': model_version,
        'run_id': run_id,
        'promoted': promoted,
        'metrics': metrics
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
        execution_timeout=timedelta(hours=12),  # Allow up to 12 hours for training (KD can take longer)
    )
    
    # Task 6: Evaluate model performance
    evaluate = PythonOperator(
        task_id='evaluate_model_performance',
        python_callable=evaluate_model_performance,
    )
    
    # Task 7: Promote to production if passing criteria
    promote = PythonOperator(
        task_id='promote_to_production',
        python_callable=promote_to_production,
    )
    
    # Task 8: Send notification
    notify = PythonOperator(
        task_id='send_training_notification',
        python_callable=send_training_notification,
    )
    
    # Task 9: Trigger TensorRT conversion for promoted model
    trigger_tensorrt = TriggerDagRunOperator(
        task_id='trigger_tensorrt_conversion',
        trigger_dag_id='convert_tensorrt',
        conf={
            'model_name': "{{ task_instance.xcom_pull(task_ids='train_model', key='model_name') }}",
            'model_version': "{{ task_instance.xcom_pull(task_ids='train_model', key='model_version') }}",
        },
        wait_for_completion=False,
        trigger_rule='all_success',  # Only trigger if promotion succeeded
    )
    
    # Define task dependencies
    fetch_data >> prepare_yaml
    fetch_teacher >> train
    prepare_yaml >> train
    prepare_config >> train
    train >> evaluate >> promote >> notify >> trigger_tensorrt
