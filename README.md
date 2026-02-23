# 🚀 AI VIETNAM MLOps Project 02

**Distillation, Quantization and TensorRT for Traffic Detection**

A complete end-to-end MLOps pipeline for object detection using YOLO, featuring knowledge distillation, model optimization, automated training, drift detection, and production monitoring.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Running Your First Demo](#-running-your-first-demo)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

This project demonstrates a production-ready MLOps pipeline for traffic detection using YOLO models. It showcases:

- **Knowledge Distillation**: Transfer knowledge from a large teacher model to a compact student model
- **Model Optimization**: Quantization and TensorRT optimization for faster inference
- **Automated Pipelines**: Airflow DAGs for training, inference, and monitoring
- **Drift Detection**: Automated data and prediction drift analysis
- **Production Serving**: FastAPI + Gradio interface with CPU and GPU support
- **Monitoring Stack**: Prometheus, Grafana, Loki, and Alertmanager for observability

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Data Pipeline│───▶│   Training   │───▶│   Serving    │     │
│  │  (Kaggle +   │    │  (Airflow +  │    │  (FastAPI +  │     │
│  │   MinIO)     │    │   MLflow)    │    │   Gradio)    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                    │                    │           │
│         ▼                    ▼                    ▼           │
│  ┌──────────────────────────────────────────────────────┐     │
│  │            Monitoring & Observability                │     │
│  │  Prometheus + Grafana + Loki + Alertmanager          │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐     │
│  │            Drift Detection & Analysis                │     │
│  │          Deepchecks + Automated Reports              │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Data Pipeline**: Kaggle integration, MinIO versioning, test variants
2. **Training Pipeline**: Knowledge distillation, MLflow tracking, model registry
3. **Serving Pipeline**: FastAPI backend, Gradio UI, CPU/GPU support
4. **Orchestration**: Airflow DAGs for automated workflows
5. **Monitoring**: Prometheus metrics, Grafana dashboards, Loki logs
6. **Quality**: Drift detection, automated alerts, model validation

---

## ✨ Features

### 🤖 Model Training & Optimization
- **Knowledge Distillation**: Train compact models with minimal accuracy loss
- **Multi-Loss Training**: Feature-based and logit-based distillation
- **MLflow Integration**: Experiment tracking, model versioning, and registry
- **Automated Pipelines**: Scheduled training and evaluation

### 📊 Data Management
- **Kaggle Integration**: Direct dataset download from Kaggle
- **Version Control**: MinIO-based dataset versioning
- **Test Variants**: Automatic generation of robustness test sets (brightness, blur, noise, fog, rain)
- **Subset Creation**: Quick demo dataset creation

### 🚀 Production Serving
- **FastAPI Backend**: RESTful API for model inference
- **Gradio UI**: User-friendly web interface
- **Dual Deployment**: CPU and GPU-accelerated endpoints
- **Health Checks**: Kubernetes-ready health and readiness probes

### 📈 Monitoring & Observability
- **Metrics Collection**: Prometheus for system and application metrics
- **Visualization**: Pre-configured Grafana dashboards
- **Log Aggregation**: Loki for centralized logging
- **Alerting**: Alertmanager with customizable alert rules
- **GPU Monitoring**: NVIDIA DCGM exporter for GPU metrics

### 🔍 Quality & Drift Detection
- **Data Drift**: Automated detection with Deepchecks
- **Prediction Drift**: Monitor model output distribution changes
- **Automated Reports**: HTML and JSON drift reports
- **Alert Integration**: Prometheus metrics for drift scores

---

## 📦 Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 20GB free space
- **GPU**: NVIDIA GPU with CUDA 11.8+ for GPU inference

### Software Dependencies

- **Docker**: 24.0+ with Docker Compose V2
- **Python**: 3.9 - 3.11
- **Git**: For repository cloning
- **Kaggle Account**: For dataset downloads (free)

---

## 🎓 Running Your First Demo

### Part 1: Environment Setup

#### 1.1 Clone and Navigate

```bash
git clone https://github.com/ThuanNaN/aio2025-mlops-project02.git
cd aio2025-mlops-project02
```

#### 1.2 Configure Environment

```bash
# Create .env file for MLflow/MinIO
cat > infra/mlflow/.env << EOF
MYSQL_DATABASE=mlflow_db
MYSQL_USER=mlflow_user
MYSQL_PASSWORD=mlflow_password
MYSQL_ROOT_PASSWORD=root_password
AWS_ACCESS_KEY_ID=minio_admin
AWS_SECRET_ACCESS_KEY=minio_password123
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
EOF
```

#### 1.3 Start Infrastructure Services

```bash
# Terminal 1: MLflow + MinIO
cd infra/mlflow
docker compose up -d
docker compose logs -f
```

```bash
# Terminal 2: Monitoring Stack
cd infra/monitor
docker compose up -d
docker compose logs -f
```

```bash
# Terminal 3: Airflow
cd infra/airflow
docker compose up -d
docker compose logs -f
```

#### 1.4 Verify Services

```bash
# Check service health
curl http://localhost:5000/health    # MLflow
curl http://localhost:9001           # MinIO Console
curl http://localhost:3000           # Grafana
curl http://localhost:8080           # Airflow (if started)

# All services should respond successfully
```

#### 1.5 Access Web Interfaces

- **MLflow**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (minio_admin / minio_password123)
- **Grafana**: http://localhost:3000 (admin / admin)
- **Airflow**: http://localhost:8080 (airflow / airflow)
- **Prometheus**: http://localhost:9090

---

### Part 2: Data Preparation

#### 2.1 Setup Kaggle Credentials

```bash
# Option 1: Environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option 2: Kaggle config file
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << EOF
{"username":"your_username","key":"your_api_key"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

**Get Kaggle API credentials**: https://www.kaggle.com/account → API → Create New Token

#### 2.2 Download Dataset

```bash
# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install data pipeline
pip install -r data_pipeline/requirements.txt

# Download full dataset
python -m data_pipeline kaggle download \
  --dataset yusufberksardoan/traffic-detection-project \
  --output data/raw \
  --organize

# Create demo subset for quick testing (recommended)
python -m data_pipeline dataset subset \
  --input data/raw \
  --output data/demo_subset \
  --size 500
```

#### 2.3 Upload to MinIO

```bash
# Version the demo dataset
python -m data_pipeline version upload \
  --dir data/demo_subset \
  --version v1.0-demo \
  --description "Demo dataset with 500 images"

# List versions
python -m data_pipeline version list
```

#### 2.4 Create Test Variants (Optional)

```bash
# Generate robustness test sets
python -m data_pipeline variants create \
  --input data/demo_subset/test \
  --output data/test_variants \
  --augmentations brightness blur noise fog

# This creates test_brightness, test_blur, test_noise, test_fog
```

---

### Part 3: Model Training

#### 3.1 Prepare Training Environment

```bash
# Install training dependencies
cd training_pipeline/src
pip install -r requirements.txt

# Verify YOLO installation
python -c "from ultralytics import YOLO; print('YOLO installed successfully')"
```

#### 3.2 Train Teacher Model (Large Model)

```bash
# Train teacher model on demo dataset
python train.py \
  --data ../../data/demo_subset/data.yaml \
  --config config/teacher_config.yaml \
  --epochs 50 \
  --batch 16 \
  --device 0  # Use GPU for CPU training

# Training takes ~10-15 minutes on GPU, ~30-45 minutes on CPU
# Monitor progress in terminal
```

#### 3.3 View Training in MLflow

```bash
# Open MLflow UI
# http://localhost:5000

# Navigate to "Experiments" → "traffic-detection"
# Compare teacher vs student metrics:
# - mAP50
# - mAP50-95
# - Training loss
# - Inference speed
```

#### 3.4 Train Student Model with Knowledge Distillation via Airflow

Go to Airflow UI (http://localhost:8080) and trigger the `train_model` DAG. This will automatically train the student model using knowledge distillation from the teacher model.


### Part 4: Inference & Serving

#### 4.1 Start Serving Pipeline

```bash
# Terminal 4: Start FastAPI + Gradio
cd serving_pipeline
docker compose up -d
docker compose logs -f
```

#### 4.2 Access Gradio Interface
Open http://localhost:7860 in your browser. You should see the Gradio interface for uploading images and getting predictions.

#### 4.3 Test Inference

**Use UI**: Upload an image from the demo dataset and click "Predict". You should see bounding boxes and class labels.

**Use Script**: Alternatively, you can test the API with the provided script to simulate drift data:

```bash
python scripts/send_drift_data.py --test-dir data/test_variants/test_brightness --api-url http://localhost:8000/predict --gpu
```

---

## 📁 Project Structure

```plantext
aio2025-mlops-project02/
│
├── data/                          # Dataset storage
│   ├── raw/                       # Original Kaggle data
│   ├── demo_subset/               # Subset for demos (500 images)
│   └── test_variants/             # Augmented test sets
│
├── data_pipeline/                 # Data management module
│   ├── __main__.py               # CLI entry point
│   ├── config.py                 # Configuration
│   ├── kaggle_download.py        # Kaggle integration
│   ├── version_manager.py        # MinIO versioning
│   ├── subset_creator.py         # Dataset subsetting
│   ├── test_variants.py          # Test augmentation
│   └── requirements.txt          # Dependencies
│
├── data_validation/               # Quality monitoring
│   ├── dataset_loader.py         # Data loading utilities
│   ├── drift_analysis.py         # Drift detection
│   ├── generate_predictions.py   # Prediction generation
│   └── requirements.txt          # Dependencies
│
├── training_pipeline/             # Model training
│   └── src/
│       ├── train.py              # Training script
│       ├── config/               # Training configs
│       │   ├── teacher_config.yaml
│       │   └── student_config.yaml
│       └── ultralytics-kd/       # Custom YOLO with KD
│           ├── distillation_loss.py
│           └── modified_trainer.py
│
├── serving_pipeline/              # Production serving
│   ├── api/                      # FastAPI backend
│   │   ├── main.py              # API entry point
│   │   ├── dependencies.py      # Shared dependencies
│   │   ├── schemas.py           # Pydantic models
│   │   └── routers/             # API routes
│   │       ├── health.py
│   │       ├── predict.py
│   │       └── batch.py
│   ├── models/                   # Model management
│   │   └── yolo_model.py        # YOLO wrapper
│   ├── utils/                    # Utilities
│   │   ├── minio_client.py      # MinIO client
│   │   └── validators.py        # Input validation
│   ├── docker/                   # Dockerfiles
│   │   ├── Dockerfile.backend
│   │   └── Dockerfile.ui
│   ├── gradio_app.py            # Gradio interface
│   ├── config.py                # Configuration
│   ├── docker-compose.yml       # Service orchestration
│   └── requirements.txt         # Dependencies
│
├── infra/                         # Infrastructure as Code
│   ├── mlflow/                   # MLflow + MinIO
│   │   ├── docker-compose.yml
│   │   └── .env.example
│   ├── airflow/                  # Airflow orchestration
│   │   ├── docker-compose.yml
│   │   ├── Dockerfile
│   │   ├── dags/                # DAG definitions
│   │   │   ├── train_model.py
│   │   │   ├── daily_inference.py
│   │   │   ├── drift_detection.py
│   │   │   └── tensorrt_conversion.py
│   │   └── config/              # Airflow configs
│   └── monitor/                  # Monitoring stack
│       ├── docker-compose.yml
│       ├── prometheus/
│       │   ├── prometheus.yml
│       │   └── alerts.yml
│       ├── grafana/
│       │   ├── datasources/
│       │   └── dashboards/
│       ├── loki/
│       │   └── loki-config.yml
│       └── alertmanager/
│           └── alertmanager.yml
│
├── scripts/                       # Utility scripts
│   ├── train_teacher_model.py    # Teacher training helper
│   ├── send_drift_data.py        # Drift data simulator
│   └── utils.ipynb               # Jupyter notebook utilities
│
├── production/                    # Production artifacts (runtime)
│   ├── images/                   # Uploaded images
│   └── predictions/              # Prediction results
│
├── reports/                       # Generated reports (runtime)
│   └── data_drift_report_*.html
│
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
