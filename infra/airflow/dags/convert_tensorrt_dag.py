"""
TensorRT Model Conversion DAG

This DAG converts promoted PyTorch models to TensorRT engines for optimized inference.
Triggered automatically after a model is promoted to Production stage.

Tasks:
1. fetch_production_model: Download production model from MLflow
2. export_tensorrt: Convert PyTorch model to TensorRT engine
3. upload_tensorrt: Upload TensorRT engine to MinIO
4. update_mlflow: Update model metadata with TensorRT info  
5. notify_conversion: Send conversion status notification

TensorRT provides 2-5x faster inference on NVIDIA GPUs with minimal accuracy loss.
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def fetch_production_model(**context):
    import mlflow
    from mlflow.tracking import MlflowClient
    from pathlib import Path
    
    print("📥 Fetching production model from MLflow...")
    dag_run_conf = context['dag_run'].conf or {}
    model_name = dag_run_conf.get('model_name', os.getenv('MODEL_NAME', 'yolo-nano-model'))
    
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minio_admin')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio_password123')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        production_versions = client.get_model_version_by_alias(model_name, "production")
        model_version = production_versions.version
        run_id = production_versions.run_id
        print(f"✅ Found model with 'production' alias: version {model_version}")
    except Exception as e:
        production_versions = client.search_model_versions(f"name='{model_name}'")
        production_models = [v for v in production_versions if v.current_stage == 'Production']
        if not production_models:
            raise ValueError(f"No production model found for {model_name}")
        latest_prod = max(production_models, key=lambda x: int(x.version))
        model_version = latest_prod.version
        run_id = latest_prod.run_id
    
    download_path = Path("/tmp/tensorrt_conversion")
    download_path.mkdir(parents=True, exist_ok=True)
    
    model_uri = f"runs:/{run_id}/weights/best.pt"
    model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(download_path))
    
    # Check if model_path is already a file or a directory
    model_path_obj = Path(model_path)
    if model_path_obj.is_file() and model_path_obj.suffix == '.pt':
        model_file = str(model_path_obj)
    else:
        # Search for .pt files in directory
        pt_files = list(model_path_obj.rglob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt file found in {model_path}")
        model_file = str(pt_files[0])
    
    print(f"✅ Model file ready: {model_file}")
    
    context['task_instance'].xcom_push(key='model_name', value=model_name)
    context['task_instance'].xcom_push(key='model_version', value=model_version)
    context['task_instance'].xcom_push(key='model_file', value=model_file)
    context['task_instance'].xcom_push(key='run_id', value=run_id)
    
    return model_file

def export_tensorrt(**context):
    from ultralytics import YOLO
    from pathlib import Path
    
    print("🔧 Converting model to TensorRT...")
    model_file = context['task_instance'].xcom_pull(task_ids='fetch_production_model', key='model_file')
    
    if not model_file or not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = YOLO(model_file)
    
    tensorrt_config = {
        'format': 'engine',
        'imgsz': int(os.getenv('TENSORRT_IMGSZ', '640')),
        'half': os.getenv('TENSORRT_FP16', 'true').lower() == 'true',
        'dynamic': os.getenv('TENSORRT_DYNAMIC', 'false').lower() == 'true',
        'workspace': float(os.getenv('TENSORRT_WORKSPACE', '4')),
        'batch': int(os.getenv('TENSORRT_BATCH', '1')),
        'device': int(os.getenv('TENSORRT_DEVICE', '0')),
    }
    
    export_path = model.export(**tensorrt_config)
    engine_size = Path(export_path).stat().st_size / (1024 * 1024)
    
    context['task_instance'].xcom_push(key='tensorrt_path', value=str(export_path))
    context['task_instance'].xcom_push(key='engine_size_mb', value=engine_size)
    
    return str(export_path)

def upload_tensorrt(**context):
    import boto3
    from botocore.client import Config
    from pathlib import Path
    from datetime import datetime
    
    tensorrt_path = context['task_instance'].xcom_pull(task_ids='export_tensorrt', key='tensorrt_path')
    model_name = context['task_instance'].xcom_pull(task_ids='fetch_production_model', key='model_name')
    model_version = context['task_instance'].xcom_pull(task_ids='fetch_production_model', key='model_version')
    
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
    BUCKET_NAME = os.getenv('TENSORRT_BUCKET', 'model-exports')
    
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minio_admin'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minio_password123'),
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
    except:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
    
    engine_filename = Path(tensorrt_path).name
    s3_key = f"tensorrt/{model_name}/v{model_version}/{engine_filename}"
    s3_key_latest = f"tensorrt/{model_name}/latest/{engine_filename}"
    
    s3_client.upload_file(tensorrt_path, BUCKET_NAME, s3_key)
    s3_client.upload_file(tensorrt_path, BUCKET_NAME, s3_key_latest)
    
    context['task_instance'].xcom_push(key='s3_key', value=s3_key)
    context['task_instance'].xcom_push(key='bucket_name', value=BUCKET_NAME)
    
    return s3_key

def update_mlflow_metadata(**context):
    import mlflow
    from mlflow.tracking import MlflowClient
    
    model_name = context['task_instance'].xcom_pull(task_ids='fetch_production_model', key='model_name')
    model_version = context['task_instance'].xcom_pull(task_ids='fetch_production_model', key='model_version')
    s3_key = context['task_instance'].xcom_pull(task_ids='upload_tensorrt', key='s3_key')
    engine_size = context['task_instance'].xcom_pull(task_ids='export_tensorrt', key='engine_size_mb')
    bucket_name = context['task_instance'].xcom_pull(task_ids='upload_tensorrt', key='bucket_name')
    
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000'))
    client = MlflowClient()
    
    model_info = client.get_model_version(model_name, model_version)
    desc = model_info.description or ""
    desc += f"\n\n🚀 TensorRT: s3://{bucket_name}/{s3_key} ({engine_size:.2f}MB)"
    
    client.update_model_version(model_name, model_version, description=desc)
    client.set_model_version_tag(model_name, model_version, "tensorrt_available", "true")
    client.set_model_version_tag(model_name, model_version, "tensorrt_path", f"s3://{bucket_name}/{s3_key}")
    
    return True

def send_conversion_notification(**context):
    model_name = context['task_instance'].xcom_pull(task_ids='fetch_production_model', key='model_name')
    model_version = context['task_instance'].xcom_pull(task_ids='fetch_production_model', key='model_version')
    s3_key = context['task_instance'].xcom_pull(task_ids='upload_tensorrt', key='s3_key')
    
    print("\n" + "="*60)
    print("🎉 TensorRT CONVERSION COMPLETE")
    print("="*60)
    print(f"📦 Model: {model_name} v{model_version}")
    print(f"🚀 Engine: {s3_key}")
    print("="*60)
    
    return {'status': 'success'}

with DAG(
    'convert_tensorrt',
    default_args=default_args,
    description='Convert production PyTorch model to TensorRT engine',
    schedule=None,
    start_date=datetime(2026, 2, 11),
    catchup=False,
    tags=['tensorrt', 'conversion', 'optimization'],
) as dag:
    
    fetch = PythonOperator(task_id='fetch_production_model', python_callable=fetch_production_model)
    export = PythonOperator(task_id='export_tensorrt', python_callable=export_tensorrt, execution_timeout=timedelta(minutes=30))
    upload = PythonOperator(task_id='upload_tensorrt', python_callable=upload_tensorrt)
    update = PythonOperator(task_id='update_mlflow_metadata', python_callable=update_mlflow_metadata)
    notify = PythonOperator(task_id='send_conversion_notification', python_callable=send_conversion_notification)
    
    fetch >> export >> upload >> update >> notify
