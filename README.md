# HPC_Nhom1_MLOps_ObjectDetection

**Tối ưu triển khai mô hình phát hiện đối tượng thời gian thực bằng nén mô hình, Docker và HPC**

Repository này là project nhóm cho học phần **Tính toán hiệu suất cao**. Mục tiêu là xây dựng lại một pipeline MLOps cho bài toán traffic object detection, trong đó mô hình YOLO được huấn luyện theo hướng teacher-student Knowledge Distillation, sau đó đóng gói thành hệ thống inference bằng FastAPI, Gradio và Docker.

## Mục Tiêu

- Huấn luyện mô hình teacher YOLO cho bài toán phát hiện đối tượng giao thông.
- Huấn luyện mô hình student nhỏ hơn bằng Knowledge Distillation.
- So sánh teacher, student baseline và student KD theo `mAP`, latency, FPS và kích thước model.
- Xuất artifact `.pt` và tùy chọn `.onnx`/TensorRT phục vụ inference.
- Triển khai API object detection bằng FastAPI.
- Xây dựng giao diện demo bằng Gradio.
- Đóng gói hệ thống bằng Docker Compose.
- Theo dõi experiment/model bằng MLflow và MinIO.
- Chuẩn bị monitoring stack với Prometheus, Grafana, Loki và Alertmanager.

## Pipeline Tổng Quan

Pipeline gốc của hệ thống production gồm data source, training, distillation, MLflow, Deepchecks, FastAPI, Gradio, monitoring và tối ưu inference bằng ONNX/TensorRT.

![Full MLOps Pipeline](assets/full_pipeline.png)

Trong điều kiện thực nghiệm của nhóm, phần training có thể được tách ra chạy offline trên Google Colab để tận dụng GPU. Sau khi train xong, các artifact như `teacher_best.pt`, `student_best.pt`, `student_kd_best.pt` và `serving_model.pt` được tải về rồi đưa vào serving pipeline.

## Kiến Trúc Chính

```text
Kaggle Dataset
  -> Data Pipeline
  -> Teacher Training
  -> Student Baseline Training
  -> Student Knowledge Distillation
  -> Model Artifacts (.pt / .onnx)
  -> FastAPI Inference Service
  -> Gradio Demo UI
  -> Docker Compose Deployment
  -> Monitoring / Drift Detection
```

## Cấu Trúc Repository

```text
hpc_nhom1_code/
├── assets/
│   └── full_pipeline.png
├── data_pipeline/
│   ├── __main__.py
│   ├── kaggle_download.py
│   ├── subset_creator.py
│   ├── test_variants.py
│   └── version_manager.py
├── data_validation/
│   ├── dataset_loader.py
│   ├── drift_analysis.py
│   └── generate_predictions.py
├── infra/
│   ├── airflow/
│   ├── mlflow/
│   └── monitor/
├── notebooks/
│   ├── colab_train.ipynb
│   └── trainer.py
├── scripts/
│   ├── train_teacher_model.py
│   └── send_drift_data.py
├── serving_pipeline/
│   ├── api/
│   ├── docker/
│   ├── models/
│   ├── utils/
│   ├── gradio_app.py
│   └── docker-compose.yml
├── training_pipeline/
│   └── src/
│       ├── train.py
│       └── config/train.yaml
└── README.md
```

## Thành Phần Chính

### Data Pipeline

Module `data_pipeline` hỗ trợ:

- tải dataset từ Kaggle;
- tổ chức dataset về cấu trúc YOLO;
- tạo subset để demo/training nhanh;
- tạo test variants như brightness, blur, noise, fog;
- upload/download version dataset qua MinIO.

### Training Pipeline

Nhóm dùng 3 nhánh training:

- **Teacher model**: YOLO lớn hơn, đóng vai trò mô hình tham chiếu.
- **Student baseline**: YOLO nhỏ hơn, train trực tiếp từ label gốc.
- **Student KD**: YOLO nhỏ hơn, học thêm từ teacher thông qua Knowledge Distillation.

Script chính:

```text
scripts/train_teacher_model.py
training_pipeline/src/train.py
notebooks/colab_train.ipynb
```

### Serving Pipeline

`serving_pipeline` cung cấp:

- FastAPI backend;
- endpoint `/detect` cho inference CPU;
- endpoint `/detect-gpu` nếu môi trường có CUDA;
- endpoint `/detect-tensorrt` nếu đã có TensorRT engine;
- Gradio UI để upload ảnh và xem bounding boxes;
- Dockerfile backend/UI;
- Docker Compose cho service API/UI.

### Infrastructure

`infra` gồm:

- `infra/mlflow`: MLflow + MinIO cho experiment tracking và artifact storage.
- `infra/airflow`: Airflow DAGs cho training, drift detection và TensorRT conversion.
- `infra/monitor`: Prometheus, Grafana, Loki và Alertmanager.

## Cài Đặt Nhanh

### 1. Clone repository và tạo venv

```bash
git clone https://github.com/hoaianthai345/HPC_Nhom1_MLOps_ObjectDectection.git
cd HPC_Nhom1_MLOps_ObjectDectection

python3 -m venv .venv-demo
source .venv-demo/bin/activate
pip install --upgrade pip
pip install -r data_pipeline/requirements.txt
pip install -r serving_pipeline/requirements.txt
pip install ultralytics mlflow pyyaml boto3 deepchecks[vision]
```

### 2. Đặt credentials

```bash
# Kaggle API token (cần để tải dataset)
cp ~/Downloads/kaggle.json secrets/
chmod 600 secrets/kaggle.json
```

`secrets/` đã được gitignore, không bao giờ commit lên GitHub. Xem `secrets/README.md` cho chi tiết.

### 3. Khởi động toàn bộ stack bằng 1 lệnh

```bash
bash scripts/start_full_local.sh
```

Script này lần lượt tạo Docker network `hpc-nhom1-network`, bật MLflow + MinIO + MySQL, monitoring stack (Prometheus + Grafana + Loki + Alertmanager), Airflow (CeleryExecutor) và serving (FastAPI + Gradio). Mỗi service được healthcheck trước khi chuyển sang service kế.

| Dịch vụ | URL | Tài khoản |
|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | — |
| Gradio UI | http://localhost:7860 | — |
| MLflow Tracking | http://localhost:5001 | — |
| MinIO Console | http://localhost:9001 | `minio_admin` / `minio_password123` |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| Alertmanager | http://localhost:9093 | — |
| Airflow Web UI | http://localhost:8080 | `airflow` / `airflow` |

Để dừng toàn bộ: `bash scripts/stop_full_local.sh`

## Hướng Dẫn Chạy Lại (Bản Final)

### Smoke test end-to-end (~10-15 phút trên CPU)

```bash
bash scripts/smoke_demo_pipeline.sh
```

Chạy đủ luồng huấn luyện-export-benchmark ở cấu hình tối giản (subset 50 ảnh, 1 epoch, imgsz 320). Mục đích là verify pipeline còn hoạt động, không phải training thật. Kết quả ở `reports/smoke_demo/`.

Các flag chính:

```bash
bash scripts/smoke_demo_pipeline.sh --no-train          # bỏ train, dùng checkpoint sẵn có
bash scripts/smoke_demo_pipeline.sh --subset 200 --epochs 10 --imgsz 416   # train kỹ hơn
bash scripts/smoke_demo_pipeline.sh --with-serving      # bật serving + test /detect
```

### Bơm traffic để Grafana có data

```bash
bash scripts/demo_warmup.sh -n 100 -i 0.3 --background
```

Bắn $N$ request vào `/detect` để Prometheus scrape có chuỗi thời gian, Grafana dashboard hiện đủ panel khi demo. Mặc định 60 request × 1s. Mở Grafana tại http://localhost:3000/d/mlops-overview với time range "Last 15 minutes" để thấy 13 panel có dữ liệu.

### Long evidence run (~8 giờ, sinh minh chứng cho báo cáo)

```bash
bash scripts/long_demo_evidence.sh                    # 8h mặc định
bash scripts/long_demo_evidence.sh --hours 4          # tuỳ chỉnh
bash scripts/long_demo_evidence.sh --skip-train       # bỏ train, chỉ chạy vòng evidence
```

Sau training pipeline, script vào vòng lặp: snapshot Prometheus/MLflow/MinIO/container health mỗi 5 phút, bơm 30 request mỗi 15 phút và gọi `/drift/data` mỗi 60 phút. Cuối cùng tự sinh `EVIDENCE_SUMMARY.md` + `timeline.csv` ở `reports/long_run_evidence/<timestamp>/`.

### Docker Swarm orchestration (Mục 3.8 báo cáo)

```bash
bash scripts/start_swarm_stack.sh        # init swarm + registry + build + deploy
bash scripts/verify_swarm_features.sh    # kiểm thử 4 tính năng vận hành
bash scripts/stop_swarm_stack.sh         # gỡ stack
```

Sau khi deploy, `mlops_api` chạy với 3 replica và rolling update `start-first`. Script verify đo bốn tính năng và lưu evidence ở `reports/swarm_evidence/<timestamp>/`: load balance qua VIP, downtime trong rolling update, thời gian self-heal sau force-kill, và Prometheus auto-discovery khi scale. Cấu hình chi tiết trong `infra/swarm/README.md`.

Lưu ý: stack swarm chiếm port 8000/7860/9090/3000 trùng compose, phải `stop_full_local.sh` (cho serving + monitor) trước khi `start_swarm_stack.sh`. MLflow + Airflow giữ ở compose vì có MySQL/Postgres stateful.

## Training Bằng Google Colab

Trong điều kiện nhóm chưa có GPU local/HPC, cách khuyến nghị là dùng notebook:

```text
notebooks/colab_train.ipynb
```

Notebook này thực hiện:

1. Clone repository project từ GitHub.
2. Clone fork `ultralytics-kd`.
3. Copy `notebooks/trainer.py` vào `ultralytics/engine/trainer.py`.
4. Cài `ultralytics-kd` ở chế độ editable.
5. Upload `kaggle.json`.
6. Tải dataset từ Kaggle.
7. Tạo subset YOLO nếu cần.
8. Train teacher YOLO.
9. Train student baseline.
10. Train student bằng Knowledge Distillation.
11. Validate model.
12. Xuất artifact `.pt` và tùy chọn `.onnx`.

Artifacts sau khi train:

```text
teacher_best.pt
student_best.pt
student_kd_best.pt
serving_model.pt
serving_model.onnx  # optional
```

`serving_model.pt` là file chính để đưa vào FastAPI/Gradio.

## Training Local

### Train teacher

```bash
python scripts/train_teacher_model.py \
  --data data/demo_subset/data.yaml \
  --model yolo11x.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --project runs/teacher \
  --name teacher_yolo \
  --no-mlflow
```

### Train student baseline

```bash
yolo detect train \
  model=yolo11n.pt \
  data=data/demo_subset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/student_baseline \
  name=student_yolo
```

### Train student với Knowledge Distillation

```bash
python training_pipeline/src/train.py \
  training_pipeline/src/config/train.yaml \
  --teacher-weights runs/teacher/teacher_yolo/weights/best.pt \
  --student-weights yolo11n.pt \
  --data data/demo_subset/data.yaml \
  --mlflow-tracking-uri http://localhost:5000 \
  --mlflow-experiment traffic-distillation \
  --mlflow-run-name student_kd
```

Lưu ý: KD cần bản `ultralytics-kd` đã được patch bằng `notebooks/trainer.py`.

## Chạy Serving Demo (chỉ stack serving)

Khi không cần đầy đủ MLflow + monitor + Airflow, có thể chỉ chạy serving:

```bash
# Copy model đã train (hoặc dùng demo model có sẵn)
bash scripts/prepare_demo_model.sh "/path/to/model.pt"

# CPU
cd serving_pipeline && docker compose up -d api ui

# GPU + TensorRT
cd serving_pipeline && docker compose --profile gpu up -d
```

Truy cập FastAPI tại http://localhost:8000/docs và Gradio tại http://localhost:7860. Drift endpoint `/drift/data` đã tích hợp Deepchecks (xem Mục 3.7 báo cáo).

## Bộ Script Tham Khảo

| Script | Mục đích | Thời lượng |
|---|---|---|
| `start_full_local.sh` | Khởi động toàn bộ stack (MLflow + Monitor + Airflow + Serving) | ~2 phút |
| `stop_full_local.sh` | Dừng toàn bộ stack | <30s |
| `smoke_demo_pipeline.sh` | Smoke test E2E train → export → serve → benchmark | 10-15 phút |
| `demo_warmup.sh` | Bơm traffic vào FastAPI để Grafana có data | tuỳ N |
| `long_demo_evidence.sh` | Chạy 8h evidence loop (snapshot + drift) cho minh chứng báo cáo | 4-8h |
| `start_swarm_stack.sh` | Init swarm + registry + deploy serving/monitor stack | ~5 phút |
| `verify_swarm_features.sh` | Kiểm thử 4 tính năng Swarm (load balance, rolling, self-heal, scale) | ~3 phút |
| `stop_swarm_stack.sh` | Gỡ swarm stack | <30s |

## Tài Liệu Cho Dev

- [docs/dev_build_matrix.md](docs/dev_build_matrix.md): hướng dẫn build theo từng loại máy.
- [docs/team_improvement_guide.md](docs/team_improvement_guide.md): phân công và hạng mục cải tiến.
- [infra/swarm/README.md](infra/swarm/README.md): chi tiết Docker Swarm deployment.
- [secrets/README.md](secrets/README.md): hướng dẫn đặt `kaggle.json` và quy tắc bảo mật.

## API Chính

### Health check

```http
GET /health
```

### Object detection

```http
POST /detect
```

Input:

- image file: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- optional query params: `confidence_threshold`, `iou_threshold`

Output:

```json
{
  "request_id": "abc123",
  "num_detections": 3,
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "inference_time_ms": 25.4,
  "detections": [
    {
      "bbox": {
        "x1": 100.0,
        "y1": 120.0,
        "x2": 300.0,
        "y2": 360.0,
        "width": 200.0,
        "height": 240.0,
        "center_x": 200.0,
        "center_y": 240.0
      },
      "confidence": 0.91,
      "class_id": 2,
      "class_name": "car"
    }
  ]
}
```



## Ghi Nhận Nguồn Tham Khảo

Repository này được nhóm chỉnh sửa và triển khai lại cho đề tài học phần từ project tham khảo:

- Original reference repository: `https://github.com/ThuanNaN/aio2025-mlops-project02`
- Nội dung tham khảo chính: MLOps pipeline cho traffic detection, data pipeline, training/serving structure, MLflow/MinIO/Airflow/monitoring setup và ý tưởng Knowledge Distillation cho YOLO.

Nhóm đã điều chỉnh lại repository theo phạm vi đề tài **HPC_Nhom1_MLOps_ObjectDetection**, bổ sung notebook Colab training, custom KD trainer, serving model wrapper, tài liệu báo cáo/slide và pipeline thực nghiệm phù hợp với điều kiện triển khai của nhóm.
