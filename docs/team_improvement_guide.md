# Hướng Dẫn Thực Hiện Các Hướng Cải Tiến

Tài liệu này dùng để hướng dẫn sau khi đã chạy được pipeline cơ bản: tải dataset, train teacher, train student baseline, train student KD và xuất `serving_model.pt`.

Mục tiêu chung là tạo đủ evidence cho báo cáo:

- Kết quả model: `mAP50`, `mAP50-95`, precision, recall.
- Kết quả hiệu năng: latency, FPS, throughput, p95 latency, memory.
- Kết quả deployment: PyTorch, ONNX, TensorRT, API, monitoring.
- Artifact: bảng CSV, biểu đồ, screenshot, model files, log train/eval.

## Quy Ước Artifact

Nên thống nhất lưu kết quả theo cấu trúc:

```text
reports/
  experiments/
    training_summary.csv
    model_eval_summary.csv
    inference_benchmark.csv
    api_load_test.csv
    drift_summary.csv
  figures/
    confusion_matrix_*.png
    results_*.png
    grafana_*.png
  logs/
    train_*.log
    eval_*.log
model_artifacts/
  teacher_best.pt
  student_best.pt
  student_kd_best.pt
  serving_model.pt
  serving_model.onnx
  serving_model.engine
```

Mỗi hướng cải tiến phải nộp tối thiểu:

- Command hoặc notebook cell đã chạy.
- File kết quả hoặc screenshot.
- Nhận xét 3-5 dòng: kết quả tăng/giảm gì, trade-off là gì.

## 1. Tăng Epochs Từ 5 Lên 30-100

Mục tiêu: kết quả `mAP` có ý nghĩa hơn, tránh báo cáo dựa trên model train quá ít epoch.

Người thực hiện cần chạy lại training với ít nhất 3 mức:

| Run | Epochs | Mục đích |
|---|---:|---|
| quick | 5 | Kiểm tra pipeline |
| medium | 30 | Kết quả báo cáo tối thiểu |
| full | 50-100 | Kết quả chính nếu đủ GPU/time |

Teacher:

```bash
python scripts/train_teacher_model.py \
  --data /content/data/demo_subset/data.yaml \
  --model yolo11x.pt \
  --epochs 50 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/teacher \
  --name teacher_yolo_e50 \
  --no-mlflow
```

Student baseline:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/content/data/demo_subset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/student_baseline \
  name=student_yolo11n_e50
```

Student KD:

```bash
python training_pipeline/src/train.py \
  training_pipeline/src/config/train.yaml \
  --teacher-weights runs/teacher/teacher_yolo_e50/weights/best.pt \
  --student-weights yolo11n.pt \
  --data /content/data/demo_subset/data.yaml \
  --mlflow-tracking-uri runs/mlflow \
  --mlflow-experiment traffic-distillation \
  --mlflow-run-name student_kd_yolo11n_e50
```

Kết quả cần ghi:

| Run | Epochs | Best epoch | mAP50 | mAP50-95 | Training time | Note |
|---|---:|---:|---:|---:|---:|---|
| teacher | 50 | TBD | TBD | TBD | TBD | TBD |
| student baseline | 50 | TBD | TBD | TBD | TBD | TBD |
| student KD | 50 | TBD | TBD | TBD | TBD | TBD |

## 2. So Sánh Nhiều Student

Mục tiêu: tìm student có trade-off tốt giữa accuracy và tốc độ.

Chạy ít nhất các model:

```text
yolo11n.pt
yolo11s.pt
yolo26n.pt  # chỉ chạy nếu fork/official đang hỗ trợ yolo26
```

Command mẫu:

```bash
for model in yolo11n.pt yolo11s.pt; do
  yolo detect train \
    model=$model \
    data=/content/data/demo_subset/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=runs/student_baseline \
    name=${model%.pt}_baseline_e50
done
```

KD với từng student:

```bash
python training_pipeline/src/train.py \
  training_pipeline/src/config/train.yaml \
  --teacher-weights runs/teacher/teacher_yolo_e50/weights/best.pt \
  --student-weights yolo11s.pt \
  --data /content/data/demo_subset/data.yaml \
  --mlflow-tracking-uri runs/mlflow \
  --mlflow-experiment traffic-distillation \
  --mlflow-run-name student_kd_yolo11s_e50
```

Bảng cần nộp:

| Student | KD | Params/Size | mAP50 | mAP50-95 | Latency ms | FPS | Nhận xét |
|---|---|---:|---:|---:|---:|---:|---|
| yolo11n | No | TBD | TBD | TBD | TBD | TBD | TBD |
| yolo11n | Yes | TBD | TBD | TBD | TBD | TBD | TBD |
| yolo11s | No | TBD | TBD | TBD | TBD | TBD | TBD |
| yolo11s | Yes | TBD | TBD | TBD | TBD | TBD | TBD |

## 3. Tune KD Loss Weight

Mục tiêu: tìm cấu hình KD tốt nhất, tránh chọn loss weight cảm tính.

File cấu hình:

```text
training_pipeline/src/config/train.yaml
```

Các tham số đang dùng:

```yaml
distillation:
  logit_temperature: 3.0
  dense_logit_weight: 0.25
  sparse_logit_weight: 0.25
  box_loss_weight: 0.5
  box_objectness_threshold: 0.3
```

Ma trận thử nghiệm đề xuất:

| Run | Temperature | Dense | Sparse | Box | Mục đích |
|---|---:|---:|---:|---:|---|
| kd_base | 3.0 | 0.25 | 0.25 | 0.50 | Baseline |
| kd_logit_high | 4.0 | 0.50 | 0.50 | 0.25 | Ưu tiên class/logit |
| kd_box_high | 3.0 | 0.15 | 0.15 | 1.00 | Ưu tiên bbox |
| kd_soft | 6.0 | 0.25 | 0.25 | 0.50 | Softer teacher |

Cách làm an toàn:

1. Copy `train.yaml` thành file mới, ví dụ `train_kd_logit_high.yaml`.
2. Chỉnh phần `distillation`.
3. Chạy KD với config đó.
4. Validate model và điền bảng.

Command:

```bash
python training_pipeline/src/train.py \
  training_pipeline/src/config/train_kd_logit_high.yaml \
  --teacher-weights runs/teacher/teacher_yolo_e50/weights/best.pt \
  --student-weights yolo11n.pt \
  --data /content/data/demo_subset/data.yaml \
  --mlflow-tracking-uri runs/mlflow \
  --mlflow-experiment traffic-distillation \
  --mlflow-run-name kd_logit_high
```

Kết quả cần nộp:

| Run | Temp | Dense | Sparse | Box | mAP50 | mAP50-95 | Latency | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| kd_base | 3.0 | 0.25 | 0.25 | 0.50 | TBD | TBD | TBD | TBD |

## 4. Tune Confidence/IoU Threshold

Mục tiêu: giảm false positive/false negative khi serving.

Chạy API với nhiều threshold:

| Config | Confidence | IoU |
|---|---:|---:|
| strict | 0.50 | 0.45 |
| default | 0.25 | 0.70 |
| recall_high | 0.15 | 0.70 |
| precision_high | 0.60 | 0.50 |

Test bằng endpoint:

```bash
curl -X POST "http://localhost:8000/detect?confidence_threshold=0.25&iou_threshold=0.70" \
  -F "file=@samples/test.jpg"
```

Nếu đánh giá thủ công, chọn 20-50 ảnh đại diện và ghi:

| Image | Confidence | IoU | TP | FP | FN | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| sample_001.jpg | 0.25 | 0.70 | TBD | TBD | TBD | TBD |

Nếu dùng validation chính quy:

```bash
yolo detect val \
  model=model_artifacts/serving_model.pt \
  data=/content/data/demo_subset/data.yaml \
  imgsz=640 \
  conf=0.25 \
  iou=0.70 \
  device=0 \
  project=runs/threshold_eval \
  name=conf025_iou070
```

## 5. Export ONNX

Mục tiêu: so sánh PyTorch vs ONNX và tạo artifact phục vụ deployment.

Export:

```bash
yolo export \
  model=model_artifacts/serving_model.pt \
  format=onnx \
  imgsz=640 \
  opset=12 \
  simplify=True
```

Validate ONNX nếu Ultralytics hỗ trợ:

```bash
yolo detect val \
  model=model_artifacts/serving_model.onnx \
  data=/content/data/demo_subset/data.yaml \
  imgsz=640 \
  device=cpu \
  project=runs/eval_onnx \
  name=serving_onnx_val
```

Benchmark cần so sánh:

| Format | Device | mAP50 | mAP50-95 | Avg latency ms | Size MB |
|---|---|---:|---:|---:|---:|
| PyTorch `.pt` | GPU | TBD | TBD | TBD | TBD |
| ONNX `.onnx` | CPU | TBD | TBD | TBD | TBD |
| ONNX `.onnx` | GPU nếu có | TBD | TBD | TBD | TBD |

## 6. TensorRT

Mục tiêu: tối ưu inference trên NVIDIA GPU.

Điều kiện:

- Máy local có NVIDIA GPU.
- Cài NVIDIA driver, CUDA, Docker NVIDIA runtime.
- TensorRT chỉ nên chạy local hoặc server GPU, không nên kỳ vọng chạy ổn trên mọi Colab runtime.

Export TensorRT trực tiếp:

```bash
yolo export \
  model=model_artifacts/serving_model.pt \
  format=engine \
  imgsz=640 \
  device=0 \
  half=True
```

Hoặc chạy DAG có sẵn:

```bash
cd infra/airflow
docker compose up -d
```

DAG liên quan:

```text
infra/airflow/dags/convert_tensorrt_dag.py
```

Serving TensorRT:

```bash
cd serving_pipeline
docker compose --profile gpu up -d
```

Endpoint:

```bash
curl -X POST "http://localhost:8000/detect-tensorrt" \
  -F "file=@samples/test.jpg"
```

Bảng cần nộp:

| Model | Format | Device | Avg latency | p95 latency | FPS | GPU memory |
|---|---|---|---:|---:|---:|---:|
| Student KD | PyTorch | GPU | TBD | TBD | TBD | TBD |
| Student KD | TensorRT FP16 | GPU | TBD | TBD | TBD | TBD |

## 7. Batch Inference

Mục tiêu: đo throughput thay vì chỉ latency đơn ảnh.

Batch inference offline bằng Python:

```python
from pathlib import Path
from time import perf_counter
from ultralytics import YOLO

model = YOLO("model_artifacts/serving_model.pt")
images = list(Path("samples").glob("*.jpg"))[:100]

start = perf_counter()
results = model.predict(
    source=[str(p) for p in images],
    imgsz=640,
    device=0,
    batch=16,
    verbose=False,
)
elapsed = perf_counter() - start

print("images:", len(images))
print("elapsed_s:", elapsed)
print("throughput_fps:", len(images) / elapsed)
```

Thử các batch:

```text
1, 4, 8, 16, 32
```

Bảng:

| Batch | Images | Total time s | FPS | GPU memory | Note |
|---:|---:|---:|---:|---:|---|
| 1 | 100 | TBD | TBD | TBD | TBD |
| 8 | 100 | TBD | TBD | TBD | TBD |
| 16 | 100 | TBD | TBD | TBD | TBD |

## 8. Benchmark Script Riêng

Mục tiêu: tự động xuất CSV cho báo cáo, tránh copy tay từ log.

Đề xuất tạo:

```text
scripts/benchmark_inference.py
```

Chức năng:

- Nhận `--model`, `--image-dir`, `--device`, `--batch`, `--warmup`, `--repeat`.
- Chạy warmup trước khi đo.
- Đo avg, p50, p95, p99 latency.
- Xuất CSV vào `reports/experiments/inference_benchmark.csv`.

Command kỳ vọng:

```bash
python scripts/benchmark_inference.py \
  --model model_artifacts/serving_model.pt \
  --image-dir samples \
  --device 0 \
  --batch 1 \
  --repeat 100 \
  --output reports/experiments/inference_benchmark.csv
```

CSV schema:

```csv
model,format,device,batch,num_images,avg_ms,p50_ms,p95_ms,p99_ms,fps,size_mb
student_kd,pt,cuda,1,100,TBD,TBD,TBD,TBD,TBD,TBD
```

## 9. MLflow Logging

Mục tiêu: lưu params, metrics, artifacts rõ ràng.

Chạy MLflow local:

```bash
cd infra/mlflow
cp .env.example .env
docker compose up -d
```

MLflow UI:

```text
http://localhost:5000
```

Khi train không dùng `--no-mlflow`:

```bash
python scripts/train_teacher_model.py \
  --data data/demo_subset/data.yaml \
  --model yolo11x.pt \
  --epochs 50 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/teacher \
  --name teacher_yolo_e50 \
  --mlflow-uri http://localhost:5000 \
  --experiment-name teacher_training
```

KD:

```bash
python training_pipeline/src/train.py \
  training_pipeline/src/config/train.yaml \
  --teacher-weights runs/teacher/teacher_yolo_e50/weights/best.pt \
  --student-weights yolo11n.pt \
  --data data/demo_subset/data.yaml \
  --mlflow-tracking-uri http://localhost:5000 \
  --mlflow-experiment traffic-distillation \
  --mlflow-run-name student_kd_e50
```

Screenshot cần nộp:

- MLflow experiment list.
- Run params.
- Metrics.
- Artifacts/weights.

## 10. Drift Detection

Mục tiêu: chứng minh phần MLOps monitoring, phát hiện data distribution thay đổi.

Chạy drift analysis:

```bash
python data_validation/drift_analysis.py \
  --data-dir data_final \
  --train-split train \
  --test-split valid \
  --output-dir reports/drift \
  --batch-size 32 \
  --img-size 640 \
  --max-samples 500
```

Qua API:

```bash
curl -X POST "http://localhost:8000/drift/data" \
  -H "Content-Type: application/json" \
  -d '{
    "data_dir": "data_final",
    "train_split": "train",
    "test_split": "valid",
    "output_dir": "reports",
    "batch_size": 32,
    "img_size": 640,
    "max_samples": 500
  }'
```

Kết quả cần nộp:

| Train split | Test split | Drift score | Passed | Report path | Nhận xét |
|---|---|---:|---|---|---|
| train | valid | TBD | TBD | TBD | TBD |
| train | production | TBD | TBD | TBD | TBD |

Prometheus metric liên quan:

```text
data_drift_score
```

## 11. API Load Test

Mục tiêu: đo p95 latency, requests/sec, error rate.

Cách đơn giản bằng `hey`:

```bash
brew install hey
```

Chạy test:

```bash
hey \
  -n 100 \
  -c 5 \
  -m POST \
  -T "multipart/form-data" \
  -D /tmp/request_body.bin \
  http://localhost:8000/detect
```

Vì multipart với `hey` hơi bất tiện, cách dễ hơn là viết script Python dùng `requests`.

Đề xuất tạo:

```text
scripts/load_test_api.py
```

Pseudo flow:

```python
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

images = list(Path("samples").glob("*.jpg"))

def send(path):
    start = time.perf_counter()
    with open(path, "rb") as f:
        r = requests.post("http://localhost:8000/detect", files={"file": f})
    latency_ms = (time.perf_counter() - start) * 1000
    return r.status_code, latency_ms

with ThreadPoolExecutor(max_workers=5) as ex:
    results = list(ex.map(send, images * 5))
```

Metric cần nộp:

| Endpoint | Requests | Concurrency | Avg ms | p95 ms | RPS | Error rate |
|---|---:|---:|---:|---:|---:|---:|
| /detect | 100 | 5 | TBD | TBD | TBD | TBD |
| /detect-gpu | 100 | 5 | TBD | TBD | TBD | TBD |
| /detect-tensorrt | 100 | 5 | TBD | TBD | TBD | TBD |

## 12. Grafana Dashboard

Mục tiêu: trực quan hóa production metrics.

Chạy monitoring:

```bash
cd infra/monitor
docker compose up -d
```

Các URL:

```text
Prometheus: http://localhost:9090
Grafana: http://localhost:3000
Alertmanager: http://localhost:9093
```

FastAPI metrics:

```text
http://localhost:8000/metrics
```

Các metric nên đưa vào dashboard:

```text
http_request_duration_seconds
http_requests_total
service_ram_mb
gpu_memory_used_mb
data_drift_score
up
```

Panel cần screenshot:

- API request rate.
- p95 API latency.
- API error rate.
- RAM usage.
- GPU memory usage.
- Data drift score.
- Service health.

PromQL mẫu:

```promql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
```

```promql
sum(rate(http_requests_total[5m]))
```

```promql
service_ram_mb
```

```promql
gpu_memory_used_mb
```

```promql
data_drift_score
```

