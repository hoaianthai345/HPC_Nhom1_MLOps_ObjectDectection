"""
Airflow DAG for Daily Production Data Inference & Pseudo-Label Generation.

This DAG:
1. Checks for new daily folders in production-data bucket (organized by date)
2. Downloads teacher model from MLflow
3. Runs teacher model inference on all images in daily folders
4. Saves pseudo-labels in training-ready format (images/ and labels/ folders)
5. Stores results in pseudo-labels bucket for student model training
6. Triggers drift detection DAG for comparison
7. Tracks processed folders to avoid reprocessing

Schedule: Daily at 3 AM
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.sdk import Variable
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from minio_utils import MinIOClient


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


# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def check_new_production_folders(**context):
    """Check for new daily folders in production-data bucket."""
    try:
        minio_client = MinIOClient()
        bucket_name = "production-data"
        
        # Ensure required buckets exist
        for bucket in ["production-data", "pseudo-labels", "drift-reports"]:
            if not minio_client.bucket_exists(bucket):
                print(f"Creating bucket: {bucket}")
                minio_client.create_bucket(bucket)
        
        # List all folders (dates) in the bucket
        all_objects = minio_client.list_objects(bucket_name, prefix='')
        
        # Extract unique date folders (format: YYYY-MM-DD)
        date_folders = set()
        for obj_key in all_objects:
            # Object keys are like: 2026-02-11/image_xxxxx.jpg
            parts = obj_key.split('/')
            if len(parts) >= 2 and parts[0]:  # Has date folder
                date_folders.add(parts[0])
        
        if not date_folders:
            print("⏳ No production data folders found")
            return 'skip_inference'
        
        # Get processed folders from previous runs (stored in XCom or variable)
        processed_folders = context.get('dag_run').conf.get('processed_folders', set())
        if isinstance(processed_folders, list):
            processed_folders = set(processed_folders)
        
        # Find unprocessed folders
        unprocessed_folders = sorted(list(date_folders - processed_folders))
        
        print(f"📊 Found {len(date_folders)} total folders")
        print(f"✅ Processed: {len(processed_folders)} folders")
        print(f"🆕 Unprocessed: {len(unprocessed_folders)} folders")
        
        if unprocessed_folders:
            print(f"Processing folders: {unprocessed_folders}")
            # Push to XCom for next tasks
            context['task_instance'].xcom_push(key='unprocessed_folders', value=unprocessed_folders)
            context['task_instance'].xcom_push(key='processed_folders', value=list(processed_folders))
            return 'download_teacher_model'
        else:
            print("⏭️  All folders already processed")
            return 'skip_inference'
            
    except Exception as e:
        print(f"❌ Error checking production folders: {e}")
        return 'skip_inference'


def download_teacher_model(**context):
    """Download teacher model from MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # MLflow configuration
    mlflow_uri = get_config("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # AWS/MinIO configuration for artifact storage
    os.environ["AWS_ACCESS_KEY_ID"] = get_config("AWS_ACCESS_KEY_ID", "minio_admin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = get_config("AWS_SECRET_ACCESS_KEY", "minio_password123")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = get_config("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    
    client = MlflowClient()
    
    # Try multiple model options in order of preference
    model_options = [
        ("yolo-teacher-model", "teacher"),
        ("yolo-nano-model", "student/production"),
    ]
    
    model_found = False
    
    for model_name, model_type in model_options:
        try:
            print(f"🔍 Trying model: {model_name} ({model_type})")
            
            # Try to get model by production alias first
            try:
                model_version = client.get_model_version_by_alias(model_name, "production")
                run_id = model_version.run_id
                print(f"✅ Found model with 'production' alias: version {model_version.version}")
            except Exception:
                # Fallback to Production stage
                versions = client.search_model_versions(f"name='{model_name}'")
                production_models = [v for v in versions if v.current_stage == 'Production']
                
                if not production_models:
                    # Try any available version
                    if versions:
                        model_version = versions[0]
                        run_id = model_version.run_id
                        print(f"⚠️  No Production model, using version {model_version.version}")
                    else:
                        raise Exception(f"No versions found for {model_name}")
                else:
                    model_version = production_models[0]
                    run_id = model_version.run_id
                    print(f"✅ Found Production model: version {model_version.version}")
            
            print(f"📦 Downloading model {model_name} v{model_version.version}")
            print(f"   Run ID: {run_id}")
            print(f"   Type: {model_type}")
            
            # Download model
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="weights/best.pt",
                dst_path="/tmp/inference_model"
            )
            
            print(f"✅ Model downloaded to {local_path}")
            
            # Push to XCom
            context['task_instance'].xcom_push(key='teacher_model_path', value=local_path)
            model_found = True
            break
            
        except Exception as e:
            print(f"⚠️  Could not load {model_name}: {e}")
            continue
    
    if not model_found:
        print("❌ No model found in MLflow, trying fallback options...")
        # Fallback to default model if available
        default_model_path = "/app/yolo11x.pt"
        if Path(default_model_path).exists():
            print(f"⚠️  Using fallback model: {default_model_path}")
            context['task_instance'].xcom_push(key='teacher_model_path', value=default_model_path)
        else:
            raise Exception("No model available for inference")


def run_daily_inference(**context):
    """Run teacher model inference on unprocessed daily folders and generate pseudo-labels for training."""
    from ultralytics import YOLO
    import tempfile
    from PIL import Image
    
    # Get data from XCom
    teacher_model_path = context['task_instance'].xcom_pull(
        task_ids='download_teacher_model',
        key='teacher_model_path'
    )
    
    unprocessed_folders = context['task_instance'].xcom_pull(
        task_ids='check_new_production_folders',
        key='unprocessed_folders'
    )
    
    if not unprocessed_folders:
        print("No folders to process")
        return
    
    # Initialize MinIO client
    minio_client = MinIOClient()
    bucket_name = "production-data"
    pseudo_labels_bucket = "pseudo-labels"
    
    # Ensure pseudo-labels bucket exists
    if not minio_client.bucket_exists(pseudo_labels_bucket):
        minio_client.create_bucket(pseudo_labels_bucket)
    
    # Load teacher model
    print(f"🤖 Loading teacher model from {teacher_model_path}")
    model = YOLO(teacher_model_path)
    
    total_processed = 0
    successfully_processed_folders = []
    
    # Process each daily folder
    for date_folder in unprocessed_folders:
        print(f"\n📅 Processing folder: {date_folder}")
        
        # List all images in this date folder
        all_objects = minio_client.list_objects(bucket_name, prefix=f"{date_folder}/")
        image_objects = [obj for obj in all_objects if obj.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   Found {len(image_objects)} images")
        
        if not image_objects:
            print(f"   ⏭️  Skipping empty folder")
            successfully_processed_folders.append(date_folder)
            continue
        
        folder_processed = 0
        
        # Process each image
        for obj_key in image_objects:
            try:
                # Download image to temp file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
                    tmp_img_path = tmp_img.name
                    minio_client.download_file(bucket_name, obj_key, tmp_img_path)
                
                # Run inference
                results = model.predict(
                    source=tmp_img_path,
                    conf=0.25,
                    iou=0.45,
                    verbose=False
                )
                
                # Generate filenames for training-ready format
                img_filename = Path(obj_key).name  # Full filename with extension
                img_stem = Path(obj_key).stem
                
                # Save in training-ready format: date_folder/images/ and date_folder/labels/
                pseudo_label_image_key = f"{date_folder}/images/{img_filename}"
                pseudo_label_txt_key = f"{date_folder}/labels/{img_stem}.txt"
                
                # Save predictions to temp file in YOLO format (without confidence scores for training)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_pred:
                    tmp_pred_path = tmp_pred.name
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Get box coordinates in normalized format
                            x_center = float(box.xywhn[0][0])
                            y_center = float(box.xywhn[0][1])
                            width = float(box.xywhn[0][2])
                            height = float(box.xywhn[0][3])
                            cls = int(box.cls[0])
                            
                            # Training format: class x_center y_center width height (no confidence)
                            tmp_pred.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # Upload image to pseudo-labels/date/images/
                minio_client.upload_file(tmp_img_path, pseudo_labels_bucket, pseudo_label_image_key)
                
                # Upload label to pseudo-labels/date/labels/
                minio_client.upload_file(tmp_pred_path, pseudo_labels_bucket, pseudo_label_txt_key)
                
                # Cleanup temp files
                Path(tmp_img_path).unlink(missing_ok=True)
                Path(tmp_pred_path).unlink(missing_ok=True)
                
                folder_processed += 1
                total_processed += 1
                
                if folder_processed % 50 == 0:
                    print(f"   Processed {folder_processed}/{len(image_objects)} images")
                
            except Exception as e:
                print(f"   ⚠️  Error processing {obj_key}: {e}")
                continue
        
        print(f"   ✅ Completed: {folder_processed}/{len(image_objects)} images processed")
        print(f"   📝 Pseudo-labels saved to: {date_folder}/images/ and {date_folder}/labels/")
        successfully_processed_folders.append(date_folder)
    
    print(f"\n🎉 Pseudo-label generation completed!")
    print(f"   Total images processed: {total_processed}")
    print(f"   Folders processed: {len(successfully_processed_folders)}")
    print(f"   📁 Dataset ready for training in 'pseudo-labels' bucket")
    
    # Push results to XCom
    context['task_instance'].xcom_push(key='total_processed', value=total_processed)
    context['task_instance'].xcom_push(key='successfully_processed_folders', value=successfully_processed_folders)


def update_processed_folders(**context):
    """Update the list of processed folders for future runs."""
    from airflow.sdk import Variable
    
    # Get previously processed folders
    processed_folders = context['task_instance'].xcom_pull(
        task_ids='check_new_production_folders',
        key='processed_folders'
    ) or []
    
    # Get newly processed folders
    newly_processed = context['task_instance'].xcom_pull(
        task_ids='run_daily_inference',
        key='successfully_processed_folders'
    ) or []
    
    # Combine and deduplicate
    all_processed = list(set(processed_folders + newly_processed))
    
    # Store in Airflow Variable for persistence across runs
    try:
        Variable.set("production_processed_folders", all_processed, serialize_json=True)
        print(f"✅ Updated processed folders: {len(all_processed)} total")
    except Exception as e:
        print(f"⚠️  Could not save processed folders: {e}")
    
    # Also push to XCom
    context['task_instance'].xcom_push(key='all_processed_folders', value=all_processed)


def skip_inference(**context):
    """Placeholder task when inference is skipped."""
    print("⏭️  Skipping daily inference (no new data)")


# Create DAG
with DAG(
    'daily_production_inference',
    default_args=default_args,
    description='Generate pseudo-labels from teacher model inference on daily production data',
    schedule='0 3 * * *',  # Run daily at 3 AM
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=['mlops', 'inference', 'production', 'daily'],
) as dag:
    
    # Check for new production folders
    check_folders = BranchPythonOperator(
        task_id='check_new_production_folders',
        python_callable=check_new_production_folders,
    )
    
    # Download teacher model
    download_model = PythonOperator(
        task_id='download_teacher_model',
        python_callable=download_teacher_model,
    )
    
    # Run daily inference
    daily_inference = PythonOperator(
        task_id='run_daily_inference',
        python_callable=run_daily_inference,
    )
    
    # Update processed folders tracking
    update_tracking = PythonOperator(
        task_id='update_processed_folders',
        python_callable=update_processed_folders,
    )
    
    # Skip task
    skip = PythonOperator(
        task_id='skip_inference',
        python_callable=skip_inference,
    )
    
    # Optional: Trigger drift detection after processing
    trigger_drift = TriggerDagRunOperator(
        task_id='trigger_drift_detection',
        trigger_dag_id='drift_detection_pipeline',
        wait_for_completion=False,
        trigger_rule='none_failed',  # Trigger even if some tasks were skipped
    )
    
    # Define task dependencies
    check_folders >> [download_model, skip]
    download_model >> daily_inference >> update_tracking >> trigger_drift
