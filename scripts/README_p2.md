# P2 — Script đánh giá local

Bốn script standalone hỗ trợ các tác vụ P2 trong plan hoàn thiện báo cáo
(ablation + INT8 + failure case + throughput). Mọi script đều chạy được trên
CPU; nếu có GPU, truyền `--device 0` (hoặc tương đương) để tăng tốc.

## Cài đặt

Dùng venv có sẵn của repo:

```bash
cd hpc_nhom1_code
.venv-demo/bin/pip install -r serving_pipeline/requirements.txt
.venv-demo/bin/pip install onnx onnxruntime matplotlib
```

Đặt biến `$PY=.venv-demo/bin/python` cho gọn lệnh ở dưới.

## 1. Validate mAP cho mọi định dạng (`p2_validate_format.py`)

Đo lại mAP của một file `.pt`, `.onnx` hoặc `.engine` để so sánh trượt do
lượng tử FP16/INT8.

```bash
$PY scripts/p2_validate_format.py \
    --model model_artifacts/serving_model.onnx \
    --data data/demo_subset/data.yaml \
    --imgsz 640 --conf 0.001 --iou 0.7 --device cpu \
    --output reports/p2/val_onnx.json
```

Đầu ra JSON: `map50, map5095, precision, recall, size_mb` + per-class mAP
nếu có. **Dùng cho P1.1 / P1.3** (validate ONNX / TensorRT + per-class).

## 2. INT8 quantization (`p2_quantize_int8.py`)

Quant động một file `.onnx` FP32 → INT8 và đo lại mAP + latency CPU cho cả
hai phiên bản.

```bash
$PY scripts/p2_quantize_int8.py \
    --onnx model_artifacts/serving_model.onnx \
    --data data/demo_subset/data.yaml \
    --val-images data/demo_subset/valid/images \
    --output-dir reports/p2/int8 \
    --imgsz 640 --iters 50
```

Đầu ra: `reports/p2/int8/serving_model_int8.onnx` + `int8_results.json`
(mAP và latency của FP32 / INT8 trên CPU). **Dùng cho P2.2**.

## 3. Failure case analysis (`p2_failure_cases.py`)

Tìm N ảnh validation có recall thấp nhất và render side-by-side
(Input | GT | Prediction).

```bash
$PY scripts/p2_failure_cases.py \
    --model model_artifacts/student_kd_best.pt \
    --val-images data/demo_subset/valid/images \
    --val-labels data/demo_subset/valid/labels \
    --output-dir reports/p2/failure_cases \
    --n 5 --iou 0.5 --device cpu --sample 150
```

Đầu ra: 5 file PNG `failure_*.png` + `failure_cases.json` (metadata).
**Dùng cho P2.3 và P1.4 (ảnh kết quả định tính).**

## 4. Benchmark local (`p2_benchmark_local.py`)

Đo mean / p50 / p95 latency với batch=1 và throughput cho nhiều batch size.

```bash
$PY scripts/p2_benchmark_local.py \
    --model model_artifacts/student_kd_best.pt \
    --val-images data/demo_subset/valid/images \
    --output-dir reports/p2/benchmark \
    --device cpu --batches 1,2,4,8 --iters 50
```

Đầu ra: `benchmark_local.json`, `throughput.csv`, `throughput_curve.png`.
**Dùng cho P2.4**.

---

## Quy trình đề xuất

1. Mở Colab, chạy hết `colab_p2_extensions.ipynb` (1.5–2h GPU, ~$0.40).
2. Tải `p2_results.zip` về `hpc_nhom1_code/reports/p2/`.
3. Đối chiếu kết quả; nếu cần đo lại một mục trên local CPU, dùng các script
   trong file này.
4. Chèn số liệu vào báo cáo theo hướng dẫn cuối notebook.
