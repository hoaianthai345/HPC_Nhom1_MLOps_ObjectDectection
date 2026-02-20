"""
Airflow DAG for Data Drift Detection.

This DAG:
1. Downloads training data from MinIO bucket
2. Downloads production data from MinIO bucket
3. Downloads Student and Teacher models from MLflow
4. Generates predictions from both models on production data
5. Checks Data Drift: Training data vs Production data (image properties)
6. Checks Prediction Drift: Student predictions vs Teacher predictions
7. If drift detected:
   - Prepares retraining dataset (combines training data + production data with teacher labels)
   - Uploads retraining dataset to MinIO
   - Triggers retrain DAG with dataset configuration
8. Saves drift reports to MinIO

The triggered training DAG receives configuration via `conf` parameter:
- data_bucket: MinIO bucket containing retraining dataset
- data_prefix: Prefix/folder path to the retraining dataset
- timestamp: Timestamp of when dataset was created
- trigger_reason: 'drift_detected'
- data_drift_passed: Whether data drift check passed
- pred_drift_passed: Whether prediction drift check passed

The training DAG can access this via: context['dag_run'].conf

Configuration (via Airflow Variables or Environment Variables):
Priority: Airflow Variable > Environment Variable > Default

MinIO/Storage:
- MINIO_ENDPOINT (default: 'http://minio:9000')
- AWS_ACCESS_KEY_ID (default: 'minio_admin')
- AWS_SECRET_ACCESS_KEY (default: 'minio_password123')
- TRAINING_DATA_BUCKET (default: 'training-data')
- TRAINING_DATA_PREFIX (default: 'train/')
- PRODUCTION_DATA_BUCKET (default: 'production-data')
- REPORTS_BUCKET (default: 'mlops-reports')

MLflow:
- MLFLOW_TRACKING_URI (default: 'http://mlflow_server:5000')

Teacher Model:
- TEACHER_MODEL_NAME (default: 'yolo-teacher-model')
- TEACHER_MODEL_ALIAS (default: 'production')
- TEACHER_MODEL_VERSION (optional: specific version number)

Drift Detection:
- MIN_DRIFT_SAMPLES (default: 50)

To set Airflow Variables via UI:
Admin > Variables > Add
Or via CLI:
airflow variables set TEACHER_MODEL_NAME my-teacher-model
airflow variables set MIN_DRIFT_SAMPLES 100

Schedule: After batch inference or manually
"""
from datetime import datetime, timedelta
import tempfile
import os

from airflow import DAG
from airflow.sdk import Variable
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
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
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def _get_minio_client():
    """Initialize MinIO client using boto3."""
    import boto3
    from botocore.client import Config
    
    endpoint = get_config("MINIO_ENDPOINT", "http://minio:9000")
    access_key = get_config("AWS_ACCESS_KEY_ID", "minio_admin")
    secret_key = get_config("AWS_SECRET_ACCESS_KEY", "minio_password123")
    
    client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    return client


def _download_s3_folder(s3_client, bucket_name: str, prefix: str, local_dir: str):
    """Download entire folder from S3/MinIO bucket."""
    from pathlib import Path
    
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    file_count = 0
    for page in pages:
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            # Skip directories
            if key.endswith('/'):
                continue
            
            # Create relative path
            relative_path = key[len(prefix):].lstrip('/')
            if not relative_path:
                continue
                
            local_file = local_path / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            s3_client.download_file(bucket_name, key, str(local_file))
            file_count += 1
    
    return file_count


def download_training_data(**context):
    """Download training data from MinIO bucket."""
    import os
    from pathlib import Path
    
    training_bucket = get_config("TRAINING_DATA_BUCKET", "training-data")
    training_prefix = get_config("TRAINING_DATA_PREFIX", "train/")
    
    s3_client = _get_minio_client()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="train_data_")
    local_train_dir = Path(temp_dir) / "train"
    
    print(f"📥 Downloading training data from MinIO...")
    print(f"   Bucket: {training_bucket}")
    print(f"   Prefix: {training_prefix}")
    print(f"   Local: {local_train_dir}")
    
    try:
        file_count = _download_s3_folder(
            s3_client, training_bucket, training_prefix, str(local_train_dir)
        )
        print(f"✅ Downloaded {file_count} training files")
        
        # Push to XCom
        context['task_instance'].xcom_push(key='train_data_dir', value=str(temp_dir))
        
    except Exception as e:
        print(f"❌ Error downloading training data: {e}")
        raise


def download_production_data(**context):
    """Download production data from MinIO bucket."""
    import os
    from pathlib import Path
    
    production_bucket = get_config("PRODUCTION_DATA_BUCKET", "production-data")
    
    s3_client = _get_minio_client()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="prod_data_")
    local_prod_dir = Path(temp_dir)
    
    print(f"📥 Downloading production data from MinIO...")
    print(f"   Bucket: {production_bucket}")
    print(f"   Local: {local_prod_dir}")
    
    try:
        # Download images
        images_count = _download_s3_folder(
            s3_client, production_bucket, "images/", str(local_prod_dir / "images")
        )
        
        # Download student predictions (saved by serving API)
        student_preds_count = _download_s3_folder(
            s3_client, production_bucket, "predictions/", str(local_prod_dir / "predictions_student")
        )
        
        # Download labels if they exist (optional)
        try:
            labels_count = _download_s3_folder(
                s3_client, production_bucket, "labels/", str(local_prod_dir / "labels")
            )
            print(f"✅ Downloaded {images_count} images, {student_preds_count} student predictions, {labels_count} labels")
        except:
            print(f"✅ Downloaded {images_count} images, {student_preds_count} student predictions (no labels)")
        
        # Check minimum samples
        min_samples = int(get_config("MIN_DRIFT_SAMPLES", "50"))
        if images_count < min_samples:
            print(f"⚠️  Not enough production samples ({images_count} < {min_samples})")
            context['task_instance'].xcom_push(key='skip_drift', value=True)
        else:
            context['task_instance'].xcom_push(key='skip_drift', value=False)
        
        # Push to XCom
        context['task_instance'].xcom_push(key='production_data_dir', value=str(temp_dir))
        context['task_instance'].xcom_push(key='student_predictions_dir', value=str(local_prod_dir / "predictions_student"))
        
    except Exception as e:
        print(f"❌ Error downloading production data: {e}")
        raise


def download_teacher_model_from_mlflow(**context):
    """Download Teacher model from MLflow (student predictions already exist from serving API)."""
    import os
    import mlflow
    from pathlib import Path
    
    mlflow_uri = get_config("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Model configuration - use alias instead of deprecated stages
    teacher_model_name = get_config("TEACHER_MODEL_NAME", "yolo-teacher-model")
    # Support both alias (modern) and version (fallback)
    teacher_model_alias = get_config("TEACHER_MODEL_ALIAS", "production")
    teacher_model_version = get_config("TEACHER_MODEL_VERSION", None)
    
    temp_dir = tempfile.mkdtemp(prefix="teacher_model_")
    
    print(f"📥 Downloading Teacher model from MLflow...")
    print(f"   MLflow URI: {mlflow_uri}")
    print(f"   Teacher: {teacher_model_name}")
    print(f"   Note: Student predictions already downloaded from serving API")
    
    try:
        # Use modern alias-based approach (not deprecated stages)
        if teacher_model_version:
            # Use specific version if provided
            teacher_model_uri = f"models:/{teacher_model_name}/{teacher_model_version}"
            print(f"   Using version: {teacher_model_version}")
        else:
            # Use alias (modern approach)
            teacher_model_uri = f"models:/{teacher_model_name}@{teacher_model_alias}"
            print(f"   Using alias: {teacher_model_alias}")
        
        teacher_local_path = Path(temp_dir) / "teacher" / "best.pt"
        teacher_local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading teacher model from: {teacher_model_uri}")
        teacher_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=teacher_model_uri,
            dst_path=str(teacher_local_path.parent)
        )
        
        print(f"✅ Teacher model downloaded successfully to: {teacher_model_path}")
        
        # Push to XCom
        context['task_instance'].xcom_push(key='teacher_model_path', value=teacher_model_path)
        context['task_instance'].xcom_push(key='models_dir', value=temp_dir)
        
    except Exception as e:
        print(f"❌ Error downloading teacher model: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if model '{teacher_model_name}' exists in MLflow")
        print(f"  2. If using alias, verify alias '{teacher_model_alias}' is set on a model version")
        print(f"  3. If using version, verify version exists")
        print(f"\nYou can set the model version explicitly with TEACHER_MODEL_VERSION env var")
        import traceback
        traceback.print_exc()
        raise


# Note: generate_student_predictions removed - serving API already saves student predictions to MinIO
# Student predictions are downloaded along with production data


def generate_teacher_predictions(**context):
    """Generate predictions from Teacher model on production data."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, '/opt/airflow')
    
    production_data_dir = Path(context['task_instance'].xcom_pull(
        task_ids='download_production_data', key='production_data_dir'
    ))
    teacher_model_path = context['task_instance'].xcom_pull(
        task_ids='download_teacher_model_from_mlflow', key='teacher_model_path'
    )
    
    images_dir = production_data_dir / "images"
    output_dir = production_data_dir / "predictions_teacher"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎓 Generating Teacher predictions...")
    print(f"   Model: {teacher_model_path}")
    print(f"   Images: {images_dir}")
    print(f"   Output: {output_dir}")
    
    try:
        from ultralytics import YOLO
        from tqdm import tqdm
        
        # COCO to dataset class mapping
        COCO_TO_DATASET = {1: 0, 5: 1, 2: 2, 3: 3, 0: 4}
        COCO_CLASSES = list(COCO_TO_DATASET.keys())
        
        model = YOLO(teacher_model_path)
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        print(f"Found {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc="Teacher predictions"):
            results = model.predict(
                source=str(img_path),
                conf=0.25,
                iou=0.45,
                imgsz=640,
                classes=COCO_CLASSES,
                verbose=False
            )
            
            result = results[0]
            output_file = output_dir / f"{img_path.stem}.txt"
            
            with open(output_file, 'w') as f:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xywhn.cpu().numpy()
                    coco_classes = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for coco_cls, box, conf in zip(coco_classes, boxes, confs):
                        dataset_cls = COCO_TO_DATASET.get(coco_cls)
                        if dataset_cls is not None:
                            x_center, y_center, width, height = box
                            f.write(f"{dataset_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.4f}\n")
        
        print(f"✅ Generated {len(image_files)} teacher predictions")
        context['task_instance'].xcom_push(key='teacher_predictions_dir', value=str(output_dir))
        
    except Exception as e:
        print(f"❌ Error generating teacher predictions: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_data_drift_detection(**context):
    """Run data drift detection between training and production data."""
    import os
    import sys
    from pathlib import Path
    from datetime import datetime
    
    sys.path.insert(0, '/opt/airflow')
    
    train_data_dir = Path(context['task_instance'].xcom_pull(
        task_ids='download_training_data', key='train_data_dir'
    ))
    production_data_dir = Path(context['task_instance'].xcom_pull(
        task_ids='download_production_data', key='production_data_dir'
    ))
    
    reports_dir = Path(tempfile.mkdtemp(prefix="reports_"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Running Data Drift Analysis...")
    print(f"   Training data: {train_data_dir}")
    print(f"   Production data: {production_data_dir}")
    
    try:
        # Import after sys.path modification
        from data_validation.drift_analysis import create_vision_data, extract_drift_metrics, DEFAULT_CLASSES, YOLOVisionData
        from data_validation.dataset_loader import YOLODataset, collate_fn
        from deepchecks.vision import Suite
        from deepchecks.vision.checks import ImagePropertyDrift
        from torch.utils.data import DataLoader
        import torch
        
        # Load training data
        print("Loading training data...")
        train_data = create_vision_data(
            data_dir=str(train_data_dir),
            split="train",
            batch_size=32,
            img_size=640,
            max_samples=500
        )
        
        # Load production data
        print("Loading production data...")
        prod_images_dir = production_data_dir / "images"
        prod_labels_dir = production_data_dir / "labels"
        
        # Create empty labels if they don't exist
        if not prod_labels_dir.exists():
            prod_labels_dir.mkdir(parents=True, exist_ok=True)
            for img_file in prod_images_dir.glob("*.jpg"):
                (prod_labels_dir / f"{img_file.stem}.txt").touch()
            for img_file in prod_images_dir.glob("*.png"):
                (prod_labels_dir / f"{img_file.stem}.txt").touch()
        
        prod_dataset = YOLODataset(
            images_dir=str(prod_images_dir),
            labels_dir=str(prod_labels_dir),
            predictions_dir=None,
            img_size=640,
            class_names=DEFAULT_CLASSES,
            max_samples=500
        )
        
        use_pin_memory = torch.cuda.is_available()
        prod_loader = DataLoader(
            prod_dataset, batch_size=32, shuffle=False,
            num_workers=4, collate_fn=collate_fn, pin_memory=use_pin_memory
        )
        
        prod_data = YOLOVisionData(
            data_loader=prod_loader,
            label_map={i: name for i, name in enumerate(DEFAULT_CLASSES)},
            task_type='object_detection'
        )
        
        # Run drift analysis
        print("Running drift checks...")
        drift_suite = Suite(
            "Data Drift Suite",
            ImagePropertyDrift().add_condition_drift_score_less_than(0.15),
        )
        
        result = drift_suite.run(train_dataset=train_data, test_dataset=prod_data)
        drift_metrics = extract_drift_metrics(result)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"data_drift_report_{timestamp}.html"
        result.save_as_html(str(report_path), as_widget=False)
        
        print(f"📈 Data Drift Metrics:")
        for prop, score in drift_metrics.items():
            print(f"   {prop}: {score:.4f}")
        
        # Upload report to MinIO
        try:
            s3_client = _get_minio_client()
            reports_bucket = get_config("REPORTS_BUCKET", "mlops-reports")
            report_key = f"drift_detection/data_drift_report_{timestamp}.html"
            
            s3_client.upload_file(str(report_path), reports_bucket, report_key)
            print(f"✅ Report uploaded to MinIO: {reports_bucket}/{report_key}")
        except Exception as e:
            print(f"⚠️  Failed to upload report to MinIO: {e}")
        
        # Push metrics to XCom
        context['task_instance'].xcom_push(key='data_drift_passed', value=result.passed())
        context['task_instance'].xcom_push(key='data_drift_metrics', value=drift_metrics)
        context['task_instance'].xcom_push(key='data_drift_report', value=str(report_path))
        
    except Exception as e:
        print(f"❌ Error in data drift detection: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_prediction_drift_detection(**context):
    """Run prediction drift detection between student and teacher predictions."""
    import os
    import sys
    from pathlib import Path
    from datetime import datetime
    
    sys.path.insert(0, '/opt/airflow')
    
    production_data_dir = Path(context['task_instance'].xcom_pull(
        task_ids='download_production_data', key='production_data_dir'
    ))
    student_preds_dir = context['task_instance'].xcom_pull(
        task_ids='download_production_data', key='student_predictions_dir'
    )
    teacher_preds_dir = context['task_instance'].xcom_pull(
        task_ids='generate_teacher_predictions', key='teacher_predictions_dir'
    )
    
    reports_dir = Path(tempfile.mkdtemp(prefix="reports_"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Running Prediction Drift Analysis...")
    print(f"   Student predictions: {student_preds_dir}")
    print(f"   Teacher predictions: {teacher_preds_dir}")
    
    try:
        from data_validation.drift_analysis import run_prediction_drift_analysis
        
        result = run_prediction_drift_analysis(
            data_dir=str(production_data_dir),
            student_predictions_dir=student_preds_dir,
            teacher_predictions_dir=teacher_preds_dir,
            output_dir=str(reports_dir),
            batch_size=32,
            img_size=640,
            max_samples=500,
            open_browser=False
        )
        
        print(f"📈 Prediction Drift Status: {'PASSED' if result['passed'] else 'FAILED'}")
        
        # Upload report to MinIO
        try:
            s3_client = _get_minio_client()
            reports_bucket = get_config("REPORTS_BUCKET", "mlops-reports")
            report_filename = Path(result['report_path']).name
            report_key = f"drift_detection/{report_filename}"
            
            s3_client.upload_file(result['report_path'], reports_bucket, report_key)
            print(f"✅ Report uploaded to MinIO: {reports_bucket}/{report_key}")
        except Exception as e:
            print(f"⚠️  Failed to upload report to MinIO: {e}")
        
        # Push metrics to XCom
        context['task_instance'].xcom_push(key='pred_drift_passed', value=result['passed'])
        context['task_instance'].xcom_push(key='pred_drift_report', value=result['report_path'])
        
    except Exception as e:
        print(f"❌ Error in prediction drift detection: {e}")
        import traceback
        traceback.print_exc()
        raise


def check_combined_drift_threshold(**context):
    """Check if drift exceeds threshold and decide whether to retrain."""
    data_drift_passed = context['task_instance'].xcom_pull(
        task_ids='run_data_drift_detection', key='data_drift_passed'
    )
    pred_drift_passed = context['task_instance'].xcom_pull(
        task_ids='run_prediction_drift_detection', key='pred_drift_passed'
    )
    data_drift_metrics = context['task_instance'].xcom_pull(
        task_ids='run_data_drift_detection', key='data_drift_metrics'
    ) or {}
    
    print(f"🔍 Checking drift thresholds:")
    print(f"   Data Drift: {'PASSED ✓' if data_drift_passed else 'FAILED ✗'}")
    print(f"   Prediction Drift: {'PASSED ✓' if pred_drift_passed else 'FAILED ✗'}")
    
    if data_drift_metrics:
        print(f"   Data drift details:")
        for prop, score in data_drift_metrics.items():
            print(f"      {prop}: {score:.4f}")
    
    # Trigger retrain if either drift check failed
    if not data_drift_passed or not pred_drift_passed:
        print("\\n🚨 DRIFT DETECTED! Triggering retrain pipeline...")
        return 'prepare_retrain_dataset'
    else:
        print("\\n✅ No significant drift detected")
        return 'skip_retrain'


def prepare_retraining_dataset(**context):
    """Prepare retraining dataset by combining training data with production data."""
    import os
    import shutil
    from pathlib import Path
    from datetime import datetime
    
    train_data_dir = Path(context['task_instance'].xcom_pull(
        task_ids='download_training_data', key='train_data_dir'
    ))
    production_data_dir = Path(context['task_instance'].xcom_pull(
        task_ids='download_production_data', key='production_data_dir'
    ))
    
    # Create new dataset directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    retrain_dir = Path(tempfile.mkdtemp(prefix="retrain_data_"))
    retrain_dataset = retrain_dir / f"retrain_{timestamp}"
    retrain_dataset.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Preparing retraining dataset...")
    print(f"   Training data: {train_data_dir}")
    print(f"   Production data: {production_data_dir}")
    print(f"   Output: {retrain_dataset}")
    
    try:
        # Create train/images and train/labels directories
        (retrain_dataset / "train" / "images").mkdir(parents=True, exist_ok=True)
        (retrain_dataset / "train" / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copy original training data
        train_images = train_data_dir / "train" / "images"
        train_labels = train_data_dir / "train" / "labels"
        
        if train_images.exists():
            for img in train_images.glob("*"):
                shutil.copy2(img, retrain_dataset / "train" / "images" / img.name)
        
        if train_labels.exists():
            for lbl in train_labels.glob("*.txt"):
                shutil.copy2(lbl, retrain_dataset / "train" / "labels" / lbl.name)
        
        # Add production data (images + teacher predictions as labels)
        prod_images = production_data_dir / "images"
        teacher_preds = Path(context['task_instance'].xcom_pull(
            task_ids='generate_teacher_predictions', key='teacher_predictions_dir'
        ))
        
        if prod_images.exists() and teacher_preds.exists():
            for img in prod_images.glob("*"):
                shutil.copy2(img, retrain_dataset / "train" / "images" / f"prod_{img.name}")
            
            for lbl in teacher_preds.glob("*.txt"):
                shutil.copy2(lbl, retrain_dataset / "train" / "labels" / f"prod_{lbl.name}")
        
        train_count = len(list((retrain_dataset / "train" / "images").glob("*")))
        print(f"✅ Prepared {train_count} training samples")
        
        # Create data.yaml
        data_yaml = retrain_dataset / "data.yaml"
        yaml_content = f"""path: {retrain_dataset}
train: train/images
val: train/images  # Use same for validation or add separate validation set

nc: 5
names: ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
"""
        data_yaml.write_text(yaml_content)
        
        # Upload dataset to MinIO
        s3_client = _get_minio_client()
        training_bucket = get_config("TRAINING_DATA_BUCKET", "training-data")
        dataset_prefix = f"retrain_{timestamp}/"
        
        print(f"☁️  Uploading retraining dataset to MinIO...")
        
        # Upload images
        for img_file in (retrain_dataset / "train" / "images").glob("*"):
            s3_key = f"{dataset_prefix}train/images/{img_file.name}"
            s3_client.upload_file(str(img_file), training_bucket, s3_key)
        
        # Upload labels
        for lbl_file in (retrain_dataset / "train" / "labels").glob("*.txt"):
            s3_key = f"{dataset_prefix}train/labels/{lbl_file.name}"
            s3_client.upload_file(str(lbl_file), training_bucket, s3_key)
        
        # Upload data.yaml
        s3_client.upload_file(str(data_yaml), training_bucket, f"{dataset_prefix}data.yaml")
        
        print(f"✅ Dataset uploaded to MinIO: {training_bucket}/{dataset_prefix}")
        
        # Push to XCom for trigger_retrain task
        context['task_instance'].xcom_push(key='retrain_dataset_bucket', value=training_bucket)
        context['task_instance'].xcom_push(key='retrain_dataset_prefix', value=dataset_prefix)
        context['task_instance'].xcom_push(key='retrain_timestamp', value=timestamp)
        
    except Exception as e:
        print(f"❌ Error preparing retraining dataset: {e}")
        import traceback
        traceback.print_exc()
        raise


def skip_retrain(**context):
    """Placeholder task when retrain is not needed."""
    print("⏭️  Skipping retrain (no significant drift detected)")


# Create DAG
with DAG(
    'drift_detection_pipeline',
    default_args=default_args,
    description='Detect data drift (train vs production) and prediction drift (student vs teacher) using MinIO and MLflow',
    schedule=None,  # Triggered manually or by other DAGs
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['mlops', 'drift', 'monitoring', 'minio', 'mlflow'],
) as dag:
    
    # Task 1: Download training data from MinIO
    download_train_data = PythonOperator(
        task_id='download_training_data',
        python_callable=download_training_data,
    )
    
    # Task 2: Download production data from MinIO
    download_prod_data = PythonOperator(
        task_id='download_production_data',
        python_callable=download_production_data,
    )
    
    # Task 3: Download teacher model from MLflow (student predictions already exist)
    download_teacher_model = PythonOperator(
        task_id='download_teacher_model_from_mlflow',
        python_callable=download_teacher_model_from_mlflow,
    )
    
    # Task 4: Generate teacher predictions (student predictions already downloaded)
    gen_teacher_preds = PythonOperator(
        task_id='generate_teacher_predictions',
        python_callable=generate_teacher_predictions,
    )
    
    # Task 5: Run data drift detection (train vs production)
    detect_data_drift = PythonOperator(
        task_id='run_data_drift_detection',
        python_callable=run_data_drift_detection,
    )
    
    # Task 6: Run prediction drift detection (student vs teacher)
    detect_pred_drift = PythonOperator(
        task_id='run_prediction_drift_detection',
        python_callable=run_prediction_drift_detection,
    )
    
    # Task 7: Check combined drift threshold
    check_threshold = BranchPythonOperator(
        task_id='check_combined_drift_threshold',
        python_callable=check_combined_drift_threshold,
    )
    
    # Task 8: Skip retrain task
    skip = PythonOperator(
        task_id='skip_retrain',
        python_callable=skip_retrain,
    )
    
    # Task 9: Prepare retraining dataset
    prepare_retrain = PythonOperator(
        task_id='prepare_retrain_dataset',
        python_callable=prepare_retraining_dataset,
    )
    
    # Task 10: Trigger retrain DAG with dataset configuration
    def _get_retrain_config(**context):
        """Get configuration for training DAG."""
        return {
            'data_bucket': context['task_instance'].xcom_pull(
                task_ids='prepare_retrain_dataset', key='retrain_dataset_bucket'
            ),
            'data_prefix': context['task_instance'].xcom_pull(
                task_ids='prepare_retrain_dataset', key='retrain_dataset_prefix'
            ),
            'timestamp': context['task_instance'].xcom_pull(
                task_ids='prepare_retrain_dataset', key='retrain_timestamp'
            ),
            'trigger_reason': 'drift_detected',
            'data_drift_passed': context['task_instance'].xcom_pull(
                task_ids='run_data_drift_detection', key='data_drift_passed'
            ),
            'pred_drift_passed': context['task_instance'].xcom_pull(
                task_ids='run_prediction_drift_detection', key='pred_drift_passed'
            ),
        }
    
    trigger_retrain = TriggerDagRunOperator(
        task_id='trigger_retrain',
        trigger_dag_id='train_model',
        wait_for_completion=False,
        conf="{{ ti.xcom_pull(task_ids='get_retrain_config') }}",
    )
    
    # Task 11: Get retrain config (helper task)
    get_config = PythonOperator(
        task_id='get_retrain_config',
        python_callable=_get_retrain_config,
    )
    
    # Define task dependencies
    # Step 1: Download training data (from MinIO)
    # Step 2: Download production data + student predictions (from MinIO)
    # Step 3: Download teacher model (from MLflow)
    # Step 4: Generate teacher predictions
    # Step 5: Run data drift (train vs production)
    # Step 6: Run prediction drift (student vs teacher)
    # Step 7: Check combined threshold
    # Step 8: Skip retrain (if no drift) OR Step 9: Prepare retraining dataset (if drift detected)
    # Step 10: Get retrain config
    # Step 11: Trigger retrain DAG
    
    # Download data and model in parallel
    [download_train_data, download_prod_data, download_teacher_model]
    
    # Generate teacher predictions after production data and teacher model are ready
    # (student predictions already downloaded with production data)
    [download_prod_data, download_teacher_model] >> gen_teacher_preds
    
    # Run data drift after train and prod data are downloaded
    [download_train_data, download_prod_data] >> detect_data_drift
    
    # Run prediction drift after teacher predictions are generated
    # (student predictions already available from download_prod_data)
    gen_teacher_preds >> detect_pred_drift
    
    # Check threshold after both drift detections complete
    [detect_data_drift, detect_pred_drift] >> check_threshold >> [skip, prepare_retrain]
    
    # Prepare retraining dataset (includes training data and teacher predictions on production data)
    prepare_retrain >> get_config >> trigger_retrain
