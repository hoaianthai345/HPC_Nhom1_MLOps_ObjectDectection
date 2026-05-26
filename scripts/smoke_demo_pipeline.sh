#!/usr/bin/env bash
#
# Smoke demo cho pipeline đầy đủ trên một máy local KHÔNG có GPU.
#
# Mục tiêu: verify pipeline chạy được end-to-end (train → KD → export → val →
# benchmark → optional serving), không phải training thật để hội tụ.
#
# Mặc định dùng cấu hình tối giản (subset 50 ảnh, 1 epoch, imgsz 320) để
# hoàn tất trong khoảng 10-15 phút trên CPU hiện đại.
#
# Usage:
#   bash scripts/smoke_demo_pipeline.sh                    # full pipeline
#   bash scripts/smoke_demo_pipeline.sh --no-train         # bỏ qua train, dùng checkpoint sẵn có
#   bash scripts/smoke_demo_pipeline.sh --with-serving     # bật stack serving + test /detect
#   bash scripts/smoke_demo_pipeline.sh --subset 100       # tăng subset
#   bash scripts/smoke_demo_pipeline.sh --epochs 3         # tăng epoch
#   bash scripts/smoke_demo_pipeline.sh --imgsz 416        # tăng imgsz
#   bash scripts/smoke_demo_pipeline.sh --help
#
# Exit code:
#   0  → mọi bước smoke pass
#   1  → có bước fail; xem output cuối + reports/smoke_demo/smoke_demo.log
#
set -uo pipefail

# ---------- Defaults ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBSET_SIZE=50
EPOCHS=1
IMGSZ=320
BATCH=2
TEACHER_MODEL="yolo26n.pt"
STUDENT_MODEL="yolo26n.pt"
DEVICE="cpu"
NO_TRAIN=0
WITH_SERVING=0
NO_KAGGLE=0
DATA_YAML=""
OUT_DIR="$REPO_ROOT/reports/smoke_demo"
LOG_FILE="$OUT_DIR/smoke_demo.log"
SAMPLE_IMG=""

# ---------- Parse flags ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --no-train)      NO_TRAIN=1; shift ;;
    --with-serving)  WITH_SERVING=1; shift ;;
    --no-kaggle)     NO_KAGGLE=1; shift ;;
    --subset)        SUBSET_SIZE="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --imgsz)         IMGSZ="$2"; shift 2 ;;
    --batch)         BATCH="$2"; shift 2 ;;
    --teacher)       TEACHER_MODEL="$2"; shift 2 ;;
    --student)       STUDENT_MODEL="$2"; shift 2 ;;
    --data)          DATA_YAML="$2"; shift 2 ;;
    --sample-image)  SAMPLE_IMG="$2"; shift 2 ;;
    -h|--help)
      awk '/^[^#]/{exit} /^#!/{next} /^#/{sub(/^# ?/,""); print}' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
: > "$LOG_FILE"

# ---------- Helpers ----------
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
fail() { log "❌  FAIL: $*"; exit 1; }
ok()   { log "✅  $*"; }
step() { echo; log "── Bước $1: $2 ──"; }
time_ms() { python3 -c "import time; print(int(time.time()*1000))"; }

PY="$REPO_ROOT/.venv-demo/bin/python"
if [ ! -x "$PY" ]; then
  PY=$(command -v python3 || command -v python || echo "")
  [ -n "$PY" ] || fail "Không tìm thấy Python (.venv-demo hoặc python3)"
  log "ℹ️  Dùng Python hệ thống: "$PY" (khuyến nghị: tạo .venv-demo trước)"
fi

START_TIME=$(time_ms)

# ============================================================
# Bước 1 — Check environment
# ============================================================
step 1 "Kiểm tra môi trường"

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  >> "$LOG_FILE" 2>&1 && ok "torch import OK" \
  || fail "PyTorch chưa cài. Cài: "$PY" -m pip install torch torchvision"

"$PY" -c "import ultralytics; print('ultralytics', ultralytics.__version__)" \
  >> "$LOG_FILE" 2>&1 && ok "ultralytics import OK" \
  || fail "Ultralytics chưa cài. Cài: "$PY" -m pip install ultralytics"

# Tự cài deps data_pipeline (kaggle, kagglehub, opencv...) nếu thiếu
if ! "$PY" -c "import kagglehub, kaggle" >> "$LOG_FILE" 2>&1; then
  if [ -f "$REPO_ROOT/data_pipeline/requirements.txt" ]; then
    log "Cài data_pipeline deps (kaggle, kagglehub, opencv-python, ...)..."
    "$PY" -m pip install -q -r "$REPO_ROOT/data_pipeline/requirements.txt" \
      >> "$LOG_FILE" 2>&1 || fail "Cài data_pipeline requirements lỗi"
    ok "Đã cài data_pipeline deps"
  else
    fail "Thiếu kagglehub/kaggle nhưng không tìm thấy data_pipeline/requirements.txt"
  fi
else
  ok "kaggle + kagglehub import OK"
fi

# Deps cho KD (training_pipeline/src/train.py) + ONNX export
EXTRA_DEPS=""
"$PY" -c "import mlflow" >> "$LOG_FILE" 2>&1 || EXTRA_DEPS="$EXTRA_DEPS mlflow"
"$PY" -c "import onnx, onnxruntime, onnxslim" >> "$LOG_FILE" 2>&1 \
  || EXTRA_DEPS="$EXTRA_DEPS onnx onnxruntime onnxslim"
if [ -n "$EXTRA_DEPS" ]; then
  log "Cài deps phụ trợ:$EXTRA_DEPS..."
  "$PY" -m pip install -q $EXTRA_DEPS >> "$LOG_FILE" 2>&1 \
    || fail "Cài $EXTRA_DEPS lỗi"
  ok "Đã cài deps phụ trợ"
fi

if [ "$NO_TRAIN" -eq 0 ]; then
  KD_FORK="$REPO_ROOT/training_pipeline/src/ultralytics-kd"
  if [ ! -d "$KD_FORK" ]; then
    log "ℹ️  Chuẩn bị ultralytics-kd fork cho Knowledge Distillation..."
    git clone --quiet https://github.com/ultralytics/ultralytics.git "$KD_FORK" \
      >> "$LOG_FILE" 2>&1 || fail "git clone ultralytics fork lỗi"
    cp "$REPO_ROOT/notebooks/trainer.py" "$KD_FORK/ultralytics/engine/trainer.py" \
      || fail "Không tìm thấy notebooks/trainer.py để patch"
    ok "Đã clone + patch ultralytics-kd"
  else
    ok "ultralytics-kd fork đã sẵn"
  fi
fi

# ============================================================
# Bước 2 — Chuẩn bị dữ liệu
# ============================================================
step 2 "Chuẩn bị dữ liệu (subset $SUBSET_SIZE ảnh)"

if [ -n "$DATA_YAML" ] && [ -f "$DATA_YAML" ]; then
  ok "Dùng dataset có sẵn: $DATA_YAML"
elif [ -d "$REPO_ROOT/data/demo_subset" ] && [ -f "$REPO_ROOT/data/demo_subset/data.yaml" ]; then
  DATA_YAML="$REPO_ROOT/data/demo_subset/data.yaml"
  ok "Reuse demo_subset hiện có"
else
  if [ "$NO_KAGGLE" -eq 1 ]; then
    fail "Không có data và --no-kaggle bật. Hoặc bỏ flag, hoặc truyền --data <path>"
  fi

  # Ưu tiên secrets/kaggle.json trong repo, fallback về ~/.kaggle/kaggle.json
  if [ -f "$REPO_ROOT/secrets/kaggle.json" ]; then
    export KAGGLE_CONFIG_DIR="$REPO_ROOT/secrets"
    chmod 600 "$REPO_ROOT/secrets/kaggle.json" 2>/dev/null || true
    log "Dùng credentials secrets/kaggle.json"
  elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
    log "Dùng credentials ~/.kaggle/kaggle.json"
  else
    fail "Không tìm thấy kaggle.json. Đặt vào secrets/kaggle.json hoặc ~/.kaggle/kaggle.json (xem secrets/README.md)"
  fi

  log "Tải dataset từ Kaggle..."
  cd "$REPO_ROOT"
  "$PY" -m data_pipeline kaggle download \
    --dataset yusufberksardoan/traffic-detection-project \
    --output data/raw --organize \
    >> "$LOG_FILE" 2>&1 \
    || fail "Tải Kaggle lỗi (xem $LOG_FILE)"
  "$PY" -m data_pipeline dataset subset \
    --input data/raw --output data/demo_subset --size "$SUBSET_SIZE" \
    >> "$LOG_FILE" 2>&1 \
    || fail "subset_creator lỗi"
  DATA_YAML="$REPO_ROOT/data/demo_subset/data.yaml"
  ok "Đã tạo data/demo_subset"
fi
[ -f "$DATA_YAML" ] || fail "data.yaml không tồn tại: $DATA_YAML"

# Tìm 1 sample image cho benchmark + serving check
if [ -z "$SAMPLE_IMG" ]; then
  for d in valid val test; do
    SAMPLE_IMG=$(find "$(dirname "$DATA_YAML")/$d/images" -type f 2>/dev/null | head -1)
    [ -n "$SAMPLE_IMG" ] && break
  done
fi
[ -n "$SAMPLE_IMG" ] && ok "Sample ảnh: $(basename "$SAMPLE_IMG")"

# ============================================================
# Bước 3 — Train teacher (skip nếu --no-train)
# ============================================================
step 3 "Huấn luyện teacher ($TEACHER_MODEL, $EPOCHS epoch, imgsz $IMGSZ)"

ARTIFACT_DIR="$OUT_DIR/artifacts"
mkdir -p "$ARTIFACT_DIR"
TEACHER_BEST="$ARTIFACT_DIR/teacher_best.pt"

if [ "$NO_TRAIN" -eq 1 ]; then
  if [ -f "$REPO_ROOT/model_artifacts/teacher_best.pt" ]; then
    cp "$REPO_ROOT/model_artifacts/teacher_best.pt" "$TEACHER_BEST"
    ok "Reuse model_artifacts/teacher_best.pt"
  else
    log "⚠️  --no-train bật nhưng không có teacher_best.pt -- bỏ qua teacher"
  fi
else
  cd "$REPO_ROOT"
  T0=$(time_ms)
  "$PY" -c "
from ultralytics import YOLO
m = YOLO('$TEACHER_MODEL')
m.train(data='$DATA_YAML', epochs=$EPOCHS, batch=$BATCH, imgsz=$IMGSZ,
        device='$DEVICE', project='$OUT_DIR/runs', name='teacher_smoke',
        save=True, plots=False, verbose=False, exist_ok=True)
" >> "$LOG_FILE" 2>&1 || fail "Train teacher lỗi (xem $LOG_FILE)"
  CAND="$OUT_DIR/runs/teacher_smoke/weights/best.pt"
  [ -f "$CAND" ] || fail "Không thấy teacher best.pt sau train"
  cp "$CAND" "$TEACHER_BEST"
  T1=$(time_ms)
  ok "Teacher train xong sau $((($T1-$T0)/1000))s"
fi

# ============================================================
# Bước 4 — Train student baseline
# ============================================================
step 4 "Huấn luyện student baseline ($STUDENT_MODEL, $EPOCHS epoch)"

STUDENT_BEST="$ARTIFACT_DIR/student_best.pt"
if [ "$NO_TRAIN" -eq 1 ] && [ -f "$REPO_ROOT/model_artifacts/student_best.pt" ]; then
  cp "$REPO_ROOT/model_artifacts/student_best.pt" "$STUDENT_BEST"
  ok "Reuse model_artifacts/student_best.pt"
elif [ "$NO_TRAIN" -eq 0 ]; then
  cd "$REPO_ROOT"
  T0=$(time_ms)
  "$PY" -c "
from ultralytics import YOLO
m = YOLO('$STUDENT_MODEL')
m.train(data='$DATA_YAML', epochs=$EPOCHS, batch=$BATCH, imgsz=$IMGSZ,
        device='$DEVICE', project='$OUT_DIR/runs', name='student_smoke',
        save=True, plots=False, verbose=False, exist_ok=True)
" >> "$LOG_FILE" 2>&1 || fail "Train student baseline lỗi"
  cp "$OUT_DIR/runs/student_smoke/weights/best.pt" "$STUDENT_BEST"
  T1=$(time_ms)
  ok "Student baseline train xong sau $((($T1-$T0)/1000))s"
fi

# ============================================================
# Bước 5 — Train student KD (cần ultralytics-kd fork installed)
# ============================================================
step 5 "Huấn luyện student với Knowledge Distillation"

KD_BEST="$ARTIFACT_DIR/student_kd_best.pt"
KD_OK=0
if [ "$NO_TRAIN" -eq 1 ]; then
  if [ -f "$REPO_ROOT/model_artifacts/student_kd_best.pt" ]; then
    cp "$REPO_ROOT/model_artifacts/student_kd_best.pt" "$KD_BEST"
    ok "Reuse model_artifacts/student_kd_best.pt"
    KD_OK=1
  else
    log "⚠️  --no-train bật và không có student_kd_best.pt -- bỏ qua KD"
  fi
else
  # Install editable fork vào venv hiện hành nếu chưa
  "$PY" -c "import ultralytics, inspect, os; \
print(inspect.getfile(ultralytics)); \
assert 'ultralytics-kd' in os.path.realpath(inspect.getfile(ultralytics))" \
    >> "$LOG_FILE" 2>&1 \
    || {
      log "Cài ultralytics-kd editable..."
      "$PY" -m pip install -e "$REPO_ROOT/training_pipeline/src/ultralytics-kd" --quiet \
        >> "$LOG_FILE" 2>&1 \
        || fail "Cài ultralytics-kd lỗi"
    }

  cd "$REPO_ROOT"
  T0=$(time_ms)
  # Auto-detect MLflow server (port 5000 hoặc 5001); fallback về file store nếu không có
  MLFLOW_URI="file:$OUT_DIR/mlruns"
  for port in 5000 5001; do
    if curl -sf -o /dev/null -w "%{http_code}" "http://localhost:$port" 2>/dev/null | grep -qE "^(200|302|403)$"; then
      MLFLOW_URI="http://localhost:$port"
      log "Phát hiện MLflow server tại $MLFLOW_URI — log run lên server thật"
      # MLflow server chạy trong docker network reference MinIO qua hostname 'minio',
      # client trên host phải thay bằng localhost:9000. Đọc credentials từ
      # infra/mlflow/.env nếu có.
      MLFLOW_ENV="$REPO_ROOT/infra/mlflow/.env"
      if [ -f "$MLFLOW_ENV" ]; then
        set -a; . "$MLFLOW_ENV"; set +a
      fi
      export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://localhost:9000}"
      export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minio_admin}"
      export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minio_password123}"
      log "S3 endpoint: $MLFLOW_S3_ENDPOINT_URL"
      break
    fi
  done

  # Sinh config YAML tạm với siêu tham số smoke (epochs/batch/imgsz/device từ flag)
  SMOKE_CFG="$OUT_DIR/train_kd_smoke.yaml"
  "$PY" - <<EOF >> "$LOG_FILE" 2>&1
import yaml
from pathlib import Path
base = yaml.safe_load(Path("training_pipeline/src/config/train.yaml").read_text())
base.setdefault("training", {}).update({
    "epochs": $EPOCHS, "batch": $BATCH, "imgsz": $IMGSZ,
    "device": "$DEVICE", "seed": 42,
})
base.setdefault("logging", {}).update({
    "project": "$OUT_DIR/runs", "name": "kd_smoke",
})
Path("$SMOKE_CFG").write_text(yaml.safe_dump(base, sort_keys=False))
EOF

  "$PY" training_pipeline/src/train.py \
    "$SMOKE_CFG" \
    --teacher-weights "$TEACHER_BEST" \
    --student-weights "$STUDENT_MODEL" \
    --data "$DATA_YAML" \
    --mlflow-tracking-uri "$MLFLOW_URI" \
    --mlflow-experiment kd-smoke \
    --mlflow-run-name "kd_smoke_$(date +%H%M%S)" \
    >> "$LOG_FILE" 2>&1 || log "⚠️  Train KD lỗi -- bỏ qua, dùng student baseline"

  CAND=$(find "$OUT_DIR/runs/kd_smoke" -name "best.pt" 2>/dev/null | head -1)
  if [ -n "$CAND" ] && [ -f "$CAND" ]; then
    cp "$CAND" "$KD_BEST"
    T1=$(time_ms)
    ok "Student KD train xong sau $((($T1-$T0)/1000))s"
    KD_OK=1
  else
    cp "$STUDENT_BEST" "$KD_BEST"
    log "ℹ️  Dùng student baseline làm KD model (KD train lỗi nhưng pipeline tiếp tục)"
  fi
fi

# ============================================================
# Bước 6 — Export sang ONNX
# ============================================================
step 6 "Export ONNX"

SERVING_PT="$ARTIFACT_DIR/serving_model.pt"
SERVING_ONNX="$ARTIFACT_DIR/serving_model.onnx"
[ -f "$KD_BEST" ] && cp "$KD_BEST" "$SERVING_PT" || cp "$STUDENT_BEST" "$SERVING_PT"

T0=$(time_ms)
# Xoá ONNX cũ (nếu có từ session trước) để export sinh file mới chính xác
rm -f "$SERVING_ONNX"
# dynamic=True để ONNX accept input shape co giãn (tránh mismatch khi val/bench dùng imgsz khác)
"$PY" -c "
from ultralytics import YOLO
YOLO('$SERVING_PT').export(format='onnx', imgsz=$IMGSZ, dynamic=True, simplify=True)
" >> "$LOG_FILE" 2>&1 || log "⚠️  Export ONNX lỗi"

# Ultralytics lưu ONNX cạnh file .pt — chỉ copy nếu vị trí khác SERVING_ONNX
EXPORTED="${SERVING_PT%.pt}.onnx"
if [ -f "$EXPORTED" ] && [ "$EXPORTED" != "$SERVING_ONNX" ]; then
  cp "$EXPORTED" "$SERVING_ONNX"
fi
[ -f "$SERVING_ONNX" ] && ok "ONNX export xong sau $((($(time_ms)-$T0)/1000))s ($(du -h "$SERVING_ONNX" | cut -f1))" \
  || log "⚠️  Không tìm thấy serving_model.onnx sau export"

# ============================================================
# Bước 7 — Validate mAP nhanh (PT + ONNX)
# ============================================================
step 7 "Validate mAP nhanh"

VAL_JSON="$OUT_DIR/val_results.json"
"$PY" <<EOF >> "$LOG_FILE" 2>&1 || log "⚠️  Val lỗi"
import json
from pathlib import Path
from ultralytics import YOLO

results = {}
for label, path in [("pt", "$SERVING_PT"), ("onnx", "$SERVING_ONNX")]:
    if not Path(path).exists():
        continue
    try:
        m = YOLO(path, task="detect")
        r = m.val(data="$DATA_YAML", imgsz=$IMGSZ, conf=0.001, iou=0.7,
                  split="val", device="$DEVICE", verbose=False, plots=False)
        results[label] = {
            "size_mb": round(Path(path).stat().st_size/1048576, 2),
            "map50":   round(float(r.box.map50), 4),
            "map5095": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4),
            "recall":  round(float(r.box.mr), 4),
        }
    except Exception as e:
        results[label] = {"error": str(e)}

Path("$VAL_JSON").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(json.dumps(results, indent=2, ensure_ascii=False))
EOF
[ -f "$VAL_JSON" ] && ok "Val xong → $VAL_JSON" || log "⚠️  Không có val_results.json"

# ============================================================
# Bước 8 — Benchmark latency CPU
# ============================================================
step 8 "Benchmark latency CPU (10 iter warmup + 30 iter đo)"

BENCH_JSON="$OUT_DIR/benchmark.json"
"$PY" <<EOF >> "$LOG_FILE" 2>&1 || log "⚠️  Benchmark lỗi"
import json, time, glob
from pathlib import Path
from ultralytics import YOLO
import numpy as np

imgs = sorted(glob.glob("$(dirname "$DATA_YAML")/valid/images/*"))[:10] or \
       sorted(glob.glob("$(dirname "$DATA_YAML")/val/images/*"))[:10]
if not imgs:
    raise SystemExit("Khong co anh val cho benchmark")

bench = {}
for label, path in [("pt", "$SERVING_PT"), ("onnx", "$SERVING_ONNX")]:
    if not Path(path).exists(): continue
    try:
        m = YOLO(path, task="detect")
        # warmup
        for i in range(10):
            m.predict(imgs[i % len(imgs)], imgsz=$IMGSZ, conf=0.25, device="cpu", verbose=False)
        # measure
        times = []
        for i in range(30):
            res = m.predict(imgs[i % len(imgs)], imgsz=$IMGSZ, conf=0.25, device="cpu", verbose=False)
            times.append(res[0].speed["inference"])
        arr = np.array(times)
        bench[label] = {
            "inf_mean_ms": round(float(arr.mean()), 2),
            "inf_p95_ms":  round(float(np.percentile(arr, 95)), 2),
            "fps":         round(1000/float(arr.mean()), 1),
        }
    except Exception as e:
        bench[label] = {"error": str(e)}

Path("$BENCH_JSON").write_text(json.dumps(bench, indent=2, ensure_ascii=False))
print(json.dumps(bench, indent=2, ensure_ascii=False))
EOF
[ -f "$BENCH_JSON" ] && ok "Benchmark xong → $BENCH_JSON"

# ============================================================
# Bước 9 — (Optional) Serving stack + test /detect
# ============================================================
if [ "$WITH_SERVING" -eq 1 ]; then
  step 9 "Test serving qua Docker"

  cp "$SERVING_PT" "$REPO_ROOT/serving_model.pt"
  cd "$REPO_ROOT"
  docker compose -f serving_pipeline/docker-compose.yml up -d api >> "$LOG_FILE" 2>&1 \
    || fail "Docker compose serving up lỗi"
  log "Đợi 20s cho FastAPI khởi động..."
  sleep 20

  if curl -s -f http://localhost:8000/health > /dev/null; then
    ok "FastAPI /health OK"
    RESP=$(curl -s -F "file=@$SAMPLE_IMG" \
      "http://localhost:8000/detect?confidence_threshold=0.25&iou_threshold=0.45")
    echo "$RESP" | tee -a "$LOG_FILE" | head -c 500
    echo
    if echo "$RESP" | grep -q "bbox\|class\|inference"; then
      ok "POST /detect trả về kết quả hợp lệ"
    else
      log "⚠️  Response không như mong đợi"
    fi
  else
    log "⚠️  FastAPI /health không phản hồi -- xem docker logs"
  fi
fi

# ============================================================
# Bước cuối — Manifest
# ============================================================
step 10 "Lưu manifest"

MANIFEST="$OUT_DIR/manifest.json"
SHA_PT=$(shasum "$SERVING_PT" 2>/dev/null | awk '{print $1}')
SHA_ONNX=$(shasum "$SERVING_ONNX" 2>/dev/null | awk '{print $1}')
TOTAL=$((($(time_ms)-$START_TIME)/1000))

cat > "$MANIFEST" <<EOF
{
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_seconds": $TOTAL,
  "config": {
    "subset_size": $SUBSET_SIZE,
    "epochs": $EPOCHS,
    "imgsz": $IMGSZ,
    "batch": $BATCH,
    "teacher_model": "$TEACHER_MODEL",
    "student_model": "$STUDENT_MODEL",
    "device": "$DEVICE"
  },
  "artifacts": {
    "serving_model.pt":   {"path": "$SERVING_PT",   "sha1": "$SHA_PT"},
    "serving_model.onnx": {"path": "$SERVING_ONNX", "sha1": "$SHA_ONNX"}
  },
  "results_files": {
    "val": "$VAL_JSON",
    "benchmark": "$BENCH_JSON",
    "log": "$LOG_FILE"
  }
}
EOF
ok "Manifest: $MANIFEST"

# ============================================================
# Tóm tắt
# ============================================================
echo
log "═════════════════════════════════════════════════════"
log "  SMOKE DEMO HOÀN TẤT trong ${TOTAL}s (~$((TOTAL/60))phút)"
log "═════════════════════════════════════════════════════"
log "  Artifact:  $ARTIFACT_DIR/"
log "  Val:       $VAL_JSON"
log "  Benchmark: $BENCH_JSON"
log "  Manifest:  $MANIFEST"
log "  Log đầy đủ: $LOG_FILE"

if [ "$WITH_SERVING" -eq 1 ]; then
  log "  Serving:   http://localhost:8000/docs"
  log "  Để tắt:    docker compose -f serving_pipeline/docker-compose.yml down"
fi
echo

exit 0
