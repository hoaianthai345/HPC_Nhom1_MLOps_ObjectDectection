# Build Và Demo Theo Từng Loại Máy

Tài liệu này chuẩn hóa cách setup để các thành viên trong nhóm chạy được project trên máy khác nhau. Chọn đúng hướng theo máy đang dùng, không cần cài toàn bộ stack nếu chỉ demo.

## 1. Ma Trận Môi Trường

| Môi trường | Dùng để làm gì | Cách khuyến nghị |
|---|---|---|
| Google Colab GPU | Train teacher/student/KD, export `.pt`/`.onnx` | `notebooks/colab_train.ipynb` |
| MacBook Apple Silicon hoặc Intel CPU | Demo nhanh FastAPI + Gradio | Local Python demo |
| Windows/Linux CPU | Demo API/UI bằng Docker hoặc local Python | Docker CPU nếu Docker ổn |
| Linux NVIDIA GPU | Serving GPU, TensorRT, benchmark | Docker GPU profile |
| Máy báo cáo/slide | Chỉ xem artifacts/kết quả | Dùng package zip từ `dist/` |

## 2. Chuẩn Bị Model

Mọi hướng demo đều cần có một file model tên:

```text
serving_model.pt
```

Nếu có file `teacher_best.pt`, `student_best.pt` hoặc `student_kd_best.pt`, chuẩn bị bằng:

```bash
bash scripts/prepare_demo_model.sh "/path/to/model.pt"
```

Ví dụ trên máy hiện tại:

```bash
bash scripts/prepare_demo_model.sh \
  "/Users/anhoaithai/Documents/AHT/2. AREAS/UEH/Kì 6/Tính toán hiệu suất cao/Project/teacher_best.pt"
```

Kết quả:

```text
serving_model.pt
model_artifacts/teacher_best.pt hoặc student_best.pt hoặc student_kd_best.pt
```

Ghi chú: teacher `yolo26x` nặng, phù hợp demo chất lượng hơn nhưng latency cao. Khi có `student_kd_best.pt`, nên dùng file đó làm `serving_model.pt`.

## 3. Hướng A - Demo Nhanh Trên Mac CPU

Dùng khi máy là MacBook và chỉ cần demo với nhóm. Không dùng Docker vì Docker trên Apple Silicon có thể kéo PyTorch Linux ARM/CUDA rất nặng.

Chạy:

```bash
bash scripts/prepare_demo_model.sh "/path/to/model.pt"
bash scripts/run_local_demo.sh
```

Mở:

```text
FastAPI docs: http://localhost:8000/docs
Gradio UI   : http://localhost:7860
```

Log:

```text
reports/logs/api.log
reports/logs/gradio.log
```

Dừng demo:

```bash
bash scripts/stop_local_demo.sh
```

Nếu cài package lỗi, thử tạo venv lại:

```bash
rm -rf .venv-demo
bash scripts/run_local_demo.sh
```

## 4. Hướng B - Docker CPU Cho Linux/Windows

Dùng khi máy có Docker tốt và muốn chạy gần production hơn.

Chuẩn bị:

```bash
bash scripts/prepare_demo_model.sh "/path/to/model.pt"
docker network create hpc-nhom1-network
```

Nếu network đã tồn tại, lệnh `docker network create` sẽ báo lỗi, có thể bỏ qua.

Chạy:

```bash
cd serving_pipeline
docker compose \
  -f docker-compose.yml \
  -f docker-compose.local-demo.yml \
  up -d --build api ui
```

Mở:

```text
FastAPI docs: http://localhost:8000/docs
Gradio UI   : http://localhost:7860
```

Xem log:

```bash
docker logs -f serving_api
docker logs -f serving_ui
```

Dừng:

```bash
cd serving_pipeline
docker compose \
  -f docker-compose.yml \
  -f docker-compose.local-demo.yml \
  down
```

Ghi chú cho Mac: nếu Docker build kéo TensorRT/PyTorch CUDA quá lâu, dùng Hướng A.

## 5. Hướng C - Linux NVIDIA GPU Và TensorRT

Dùng khi có máy Linux với NVIDIA GPU. Không khuyến nghị cho Mac.

Kiểm tra GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Chạy API GPU:

```bash
cd serving_pipeline
docker compose --profile gpu up -d --build api-gpu ui
```

Endpoint:

```text
GPU API      : http://localhost:8001/docs
Gradio UI    : http://localhost:7860
```

Export TensorRT:

```bash
yolo export \
  model=model_artifacts/student_kd_best.pt \
  format=engine \
  imgsz=640 \
  device=0 \
  half=True
```

Sau khi có `.engine`, có thể cấu hình:

```bash
export YOLO_MODEL_PATH=/path/to/model.engine
export TENSORRT_ENABLED=true
```

## 6. Hướng D - Colab Training Và Xuất Artifact

Dùng để train, không dùng để demo production lâu dài.

Mở:

```text
notebooks/colab_train.ipynb
```

Notebook sẽ làm:

```text
clone repo
download Kaggle dataset
train teacher yolo26x
train student yolo26n
train KD
validate
export artifacts
```

Sau khi train, tải về:

```text
/content/model_artifacts/teacher_best.pt
/content/model_artifacts/student_best.pt
/content/model_artifacts/student_kd_best.pt
/content/model_artifacts/serving_model.pt
/content/model_artifacts/serving_model.onnx
```

Đưa về local rồi chạy:

```bash
bash scripts/prepare_demo_model.sh "/path/to/student_kd_best.pt"
bash scripts/run_local_demo.sh
```

## 7. Đóng Gói Kết Quả Cho Người Khác

Sau khi đã có docs, notebook, report, model artifacts:

```bash
bash scripts/package_project_artifacts.sh
```

File zip nằm trong:

```text
dist/
```

Gửi cho thành viên khác:

```text
dist/hpc_nhom1_artifacts_YYYYMMDD_HHMMSS.zip
```

## 8. Checklist Cho Dev Nhận Việc

Người nhận project chỉ cần làm theo thứ tự:

```bash
git pull
bash scripts/prepare_demo_model.sh "/path/to/model.pt"
bash scripts/run_local_demo.sh
```

Sau đó mở:

```text
http://localhost:7860
```

Nếu chỉ kiểm API:

```bash
curl http://localhost:8000/health
```

Nếu cần dừng:

```bash
bash scripts/stop_local_demo.sh
```

## 9. Lỗi Thường Gặp

### Docker build quá lâu trên Mac

Nguyên nhân: Docker Linux ARM kéo PyTorch/TensorRT/CUDA packages rất nặng.

Cách xử lý: dùng local Python demo:

```bash
bash scripts/run_local_demo.sh
```

### Không tìm thấy model

Chạy:

```bash
ls -lh serving_model.pt
```

Nếu chưa có:

```bash
bash scripts/prepare_demo_model.sh "/path/to/model.pt"
```

### Gradio mở nhưng detect lỗi

Xem API log:

```bash
tail -100 reports/logs/api.log
```

Thường gặp:

- Model không đúng path.
- Model quá nặng, CPU inference chậm.
- Thiếu dependency trong venv.

### Port 8000 hoặc 7860 bị chiếm

Dừng demo cũ:

```bash
bash scripts/stop_local_demo.sh
```

Hoặc đổi port:

```bash
API_PORT=8010 GRADIO_PORT=7861 bash scripts/run_local_demo.sh
```
