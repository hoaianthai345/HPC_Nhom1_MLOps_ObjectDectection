#!/usr/bin/env bash
#
# Long-running demo pipeline (mặc định ~8 giờ) để thu thập bằng chứng end-to-end
# cho báo cáo. Khác với smoke_demo_pipeline.sh (chỉ verify chạy được trong
# 10-15 phút), script này:
#   - Train 3 mô hình ở cấu hình "thật" hơn (subset lớn hơn, nhiều epoch hơn)
#   - Sau đó chạy vòng lặp evidence trong nhiều giờ:
#       * snapshot Prometheus / MLflow / MinIO / container health mỗi 5 phút
#       * bắn traffic vào FastAPI mỗi 15 phút (giữ Grafana có data liên tục)
#       * chạy drift detection mỗi 60 phút (nếu module có)
#   - Cuối cùng sinh báo cáo evidence dạng Markdown + timeline CSV
#
# Yêu cầu trước khi chạy:
#   - Stack MLflow + Monitoring + Serving đã lên (chạy: bash scripts/start_full_local.sh)
#   - secrets/kaggle.json đã có nếu muốn lấy dataset mới (không bắt buộc khi --skip-train)
#
# Usage:
#   bash scripts/long_demo_evidence.sh                     # 8 giờ mặc định
#   bash scripts/long_demo_evidence.sh --hours 4           # 4 giờ
#   bash scripts/long_demo_evidence.sh --skip-train        # bỏ qua training, chỉ chạy vòng evidence
#   bash scripts/long_demo_evidence.sh --evidence-only     # alias cho --skip-train
#   bash scripts/long_demo_evidence.sh --subset 200 --epochs 10 --imgsz 416
#   bash scripts/long_demo_evidence.sh --snapshot-interval 300 --traffic-interval 900
#   bash scripts/long_demo_evidence.sh --help
#
# Exit code:
#   0  → hoàn tất đủ thời lượng dự kiến
#   1  → có lỗi nghiêm trọng (stack chưa lên, training fail)
#   2  → user dừng (Ctrl+C) — evidence đã có sẽ vẫn được summary
#
set -uo pipefail

# ---------- Defaults ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOURS=8
SUBSET_SIZE=200
EPOCHS=10
IMGSZ=416
BATCH=4
DEVICE="cpu"
SKIP_TRAIN=0
SNAPSHOT_INTERVAL=300   # 5 min
TRAFFIC_INTERVAL=900    # 15 min
DRIFT_INTERVAL=3600     # 60 min
TRAFFIC_BURST=30
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/reports/long_run_evidence/$TIMESTAMP"
LOG_FILE=""    # sẽ set sau khi mkdir
SAMPLE_IMG=""

API_URL="http://localhost:8000"
MLFLOW_URL="http://localhost:5001"
MINIO_URL="http://localhost:9000"
PROM_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"

# ---------- Parse flags ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --hours)                HOURS="$2"; shift 2 ;;
    --subset)               SUBSET_SIZE="$2"; shift 2 ;;
    --epochs)               EPOCHS="$2"; shift 2 ;;
    --imgsz)                IMGSZ="$2"; shift 2 ;;
    --batch)                BATCH="$2"; shift 2 ;;
    --skip-train|--evidence-only) SKIP_TRAIN=1; shift ;;
    --snapshot-interval)    SNAPSHOT_INTERVAL="$2"; shift 2 ;;
    --traffic-interval)     TRAFFIC_INTERVAL="$2"; shift 2 ;;
    --drift-interval)       DRIFT_INTERVAL="$2"; shift 2 ;;
    --traffic-burst)        TRAFFIC_BURST="$2"; shift 2 ;;
    --sample-image)         SAMPLE_IMG="$2"; shift 2 ;;
    -h|--help)
      awk '/^[^#]/{exit} /^#!/{next} /^#/{sub(/^# ?/,""); print}' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR/snapshots" "$OUT_DIR/training" "$OUT_DIR/export" "$OUT_DIR/drift"
LOG_FILE="$OUT_DIR/run.log"
: > "$LOG_FILE"

TOTAL_SECONDS=$(( HOURS * 3600 ))
START_TS=$(date +%s)
DEADLINE=$(( START_TS + TOTAL_SECONDS ))

# ---------- Helpers ----------
log()   { echo "[$(date +'%F %T')] $*" | tee -a "$LOG_FILE"; }
ok()    { log "✅  $*"; }
warn()  { log "⚠️   $*"; }
fail()  { log "❌  FAIL: $*"; exit 1; }
phase() { echo | tee -a "$LOG_FILE"; log "═══ $* ═══"; }
elapsed_human() {
  local s=$1
  printf "%dh%02dm%02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

cleanup() {
  warn "Đang dừng (signal nhận được)"
  build_summary
  exit 2
}
trap cleanup INT TERM

# ---------- Phase 1: Preflight ----------
phase "Phase 1/5 — Preflight"
log "Output dir: $OUT_DIR"
log "Thời lượng dự kiến: ${HOURS}h (deadline = $(date -r "$DEADLINE" +'%F %T'))"

check_url() {
  local url="$1" name="$2"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$url" 2>/dev/null || echo "TIMEOUT")
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    ok "$name OK ($code) — $url"
    return 0
  else
    warn "$name không sẵn sàng ($code) — $url"
    return 1
  fi
}

PREFLIGHT_OK=1
check_url "$API_URL/health"   "FastAPI"     || PREFLIGHT_OK=0
check_url "$MLFLOW_URL"       "MLflow"      || PREFLIGHT_OK=0
check_url "$PROM_URL/-/healthy" "Prometheus" || PREFLIGHT_OK=0
check_url "$GRAFANA_URL/api/health" "Grafana" || PREFLIGHT_OK=0
check_url "$MINIO_URL/minio/health/live" "MinIO" || PREFLIGHT_OK=0

if [ "$PREFLIGHT_OK" -eq 0 ]; then
  fail "Stack chưa lên đầy đủ. Chạy: bash scripts/start_full_local.sh trước, rồi rerun."
fi

# Auto-detect sample image
if [ -z "$SAMPLE_IMG" ]; then
  for d in "$REPO_ROOT/data/demo_subset/valid/images" \
           "$REPO_ROOT/data/demo_subset/test/images"  \
           "$REPO_ROOT/data/raw/test/images"; do
    if [ -d "$d" ]; then
      SAMPLE_IMG=$(find "$d" -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | head -1)
      [ -n "$SAMPLE_IMG" ] && break
    fi
  done
fi
[ -n "$SAMPLE_IMG" ] && [ -f "$SAMPLE_IMG" ] \
  && ok "Sample image: $(basename "$SAMPLE_IMG")" \
  || warn "Không tìm thấy sample image. Traffic loop có thể skip."

# ---------- Phase 2: Training ----------
if [ "$SKIP_TRAIN" -eq 1 ]; then
  phase "Phase 2/5 — Training (BỎ QUA do --skip-train)"
else
  phase "Phase 2/5 — Training pipeline (teacher → student baseline → student KD)"
  log "Cấu hình: subset=$SUBSET_SIZE  epochs=$EPOCHS  imgsz=$IMGSZ  batch=$BATCH  device=$DEVICE"

  TRAIN_LOG="$OUT_DIR/training/smoke_pipeline.log"
  log "Delegating sang scripts/smoke_demo_pipeline.sh với cấu hình lớn hơn"
  log "Output → $TRAIN_LOG"
  if bash "$REPO_ROOT/scripts/smoke_demo_pipeline.sh" \
        --subset "$SUBSET_SIZE" \
        --epochs "$EPOCHS" \
        --imgsz "$IMGSZ" \
        --batch "$BATCH" \
        ${SAMPLE_IMG:+--sample-image "$SAMPLE_IMG"} \
        >> "$TRAIN_LOG" 2>&1; then
    ok "Training pipeline hoàn tất. Xem $TRAIN_LOG"
    # Copy artifacts ra cây evidence
    if [ -d "$REPO_ROOT/reports/smoke_demo" ]; then
      cp -r "$REPO_ROOT/reports/smoke_demo/"*.json "$OUT_DIR/training/" 2>/dev/null || true
      cp -r "$REPO_ROOT/reports/smoke_demo/"*.txt  "$OUT_DIR/training/" 2>/dev/null || true
    fi
  else
    warn "Training pipeline trả về non-zero. Tiếp tục Phase 3 với checkpoint sẵn có."
  fi
fi

# ---------- Phase 3: Export evidence snapshot baseline ----------
phase "Phase 3/5 — Snapshot baseline trước khi vào vòng evidence"
snapshot_now() {
  local label="$1"
  local ts; ts=$(date +'%Y%m%d_%H%M%S')
  local prefix="$OUT_DIR/snapshots/${ts}_${label}"

  curl -s "$PROM_URL/api/v1/query?query=up"              > "${prefix}_prom_up.json"        2>/dev/null || true
  curl -s "$PROM_URL/api/v1/query?query=http_requests_total" > "${prefix}_prom_http.json"   2>/dev/null || true
  curl -s "$PROM_URL/api/v1/query?query=process_resident_memory_bytes" > "${prefix}_prom_mem.json" 2>/dev/null || true

  curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/experiments/search" \
       -H 'Content-Type: application/json' -d '{"max_results":50}' \
       > "${prefix}_mlflow_experiments.json" 2>/dev/null || true
  curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/runs/search" \
       -H 'Content-Type: application/json' \
       -d '{"experiment_ids":["1"],"max_results":20,"order_by":["start_time DESC"]}' \
       > "${prefix}_mlflow_runs.json" 2>/dev/null || true

  docker ps --format '{{.Names}}|{{.Status}}' > "${prefix}_containers.txt" 2>/dev/null || true
  docker exec hpc_nhom1_minio mc ls local/ 2>/dev/null > "${prefix}_minio_buckets.txt" || true

  echo "${prefix}"
}
SNAP=$(snapshot_now baseline)
ok "Baseline snapshot tại: $(basename "$SNAP")*"

# ---------- Phase 4: Long-running evidence loop ----------
phase "Phase 4/5 — Evidence loop (snapshot ${SNAPSHOT_INTERVAL}s / traffic ${TRAFFIC_INTERVAL}s / drift ${DRIFT_INTERVAL}s)"

TIMELINE_CSV="$OUT_DIR/timeline.csv"
echo "timestamp,elapsed_sec,n_mlflow_runs,http_requests_total,fastapi_up,prometheus_up,grafana_up,minio_up" > "$TIMELINE_CSV"

LAST_TRAFFIC=0
LAST_DRIFT=0
LAST_SNAPSHOT=0
ITER=0

write_timeline_row() {
  local now=$(date +%s)
  local elapsed=$(( now - START_TS ))
  local n_runs total_req fa_up prom_up gr_up mn_up

  n_runs=$(curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/runs/search" \
            -H 'Content-Type: application/json' \
            -d '{"experiment_ids":["1"],"max_results":1000}' 2>/dev/null \
            | python3 -c "import sys,json
try: print(len(json.load(sys.stdin).get('runs',[])))
except: print(0)" 2>/dev/null || echo 0)
  total_req=$(curl -s "$PROM_URL/api/v1/query?query=sum(http_requests_total)" 2>/dev/null \
            | python3 -c "import sys,json
try:
  d=json.load(sys.stdin)['data']['result']
  print(int(float(d[0]['value'][1])) if d else 0)
except: print(0)" 2>/dev/null || echo 0)
  fa_up=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$API_URL/health" 2>/dev/null || echo 0)
  prom_up=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$PROM_URL/-/healthy" 2>/dev/null || echo 0)
  gr_up=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$GRAFANA_URL/api/health" 2>/dev/null || echo 0)
  mn_up=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$MINIO_URL/minio/health/live" 2>/dev/null || echo 0)

  echo "$(date +'%F %T'),$elapsed,$n_runs,$total_req,$fa_up,$prom_up,$gr_up,$mn_up" >> "$TIMELINE_CSV"
}

burst_traffic() {
  [ -z "$SAMPLE_IMG" ] && return 0
  local ok_count=0 fail_count=0
  for i in $(seq 1 "$TRAFFIC_BURST"); do
    if curl -sf -o /dev/null -F "file=@${SAMPLE_IMG}" \
         "$API_URL/detect?confidence_threshold=0.25&iou_threshold=0.45" 2>/dev/null; then
      ok_count=$((ok_count+1))
    else
      fail_count=$((fail_count+1))
    fi
  done
  log "🚦  Traffic burst: $ok_count ok / $fail_count fail (n=$TRAFFIC_BURST)"
}

run_drift() {
  local ts; ts=$(date +'%Y%m%d_%H%M%S')
  local meta="$OUT_DIR/drift/${ts}_drift.json"
  local html="$OUT_DIR/drift/${ts}_drift.html"
  local q="format=json&data_dir=/app/data/raw&train_split=train&test_split=valid&max_samples=50&batch_size=8&img_size=320&output_dir=/tmp"
  local code
  code=$(curl -s --max-time 240 -o "$meta" -w '%{http_code}' "$API_URL/drift/data?$q" 2>/dev/null)
  if [ "$code" = "200" ]; then
    # Pull HTML report from container's /tmp ra cây evidence
    local report_path
    report_path=$(python3 -c "import json,sys;d=json.load(open('$meta'));print(d.get('report_path',''))" 2>/dev/null)
    if [ -n "$report_path" ]; then
      docker cp "serving_api:$report_path" "$html" >/dev/null 2>&1 && \
        ok "Drift report HTML: $(basename "$html") + meta JSON" || \
        warn "Drift JSON OK nhưng không copy được HTML ($report_path)"
    else
      ok "Drift report JSON: $(basename "$meta")"
    fi
  else
    warn "Drift endpoint trả HTTP $code (bỏ qua chu kỳ này; xem $meta)"
  fi
}

log "Vòng lặp bắt đầu. Dừng sớm bằng Ctrl+C."

while :; do
  NOW=$(date +%s)
  ELAPSED=$(( NOW - START_TS ))
  REMAIN=$(( DEADLINE - NOW ))

  if [ "$REMAIN" -le 0 ]; then
    ok "Đã đạt deadline ${HOURS}h. Thoát vòng lặp."
    break
  fi

  ITER=$((ITER+1))

  # Snapshot
  if [ $(( NOW - LAST_SNAPSHOT )) -ge "$SNAPSHOT_INTERVAL" ]; then
    snapshot_now "iter$(printf '%04d' $ITER)" >/dev/null
    write_timeline_row
    log "📸 Snapshot #$ITER  |  elapsed=$(elapsed_human $ELAPSED)  |  còn=$(elapsed_human $REMAIN)"
    LAST_SNAPSHOT=$NOW
  fi

  # Traffic burst
  if [ $(( NOW - LAST_TRAFFIC )) -ge "$TRAFFIC_INTERVAL" ]; then
    burst_traffic
    LAST_TRAFFIC=$NOW
  fi

  # Drift detection
  if [ $(( NOW - LAST_DRIFT )) -ge "$DRIFT_INTERVAL" ] && [ "$LAST_DRIFT" -ne 0 ] || \
     [ "$LAST_DRIFT" -eq 0 ] && [ "$ELAPSED" -ge 60 ]; then
    # chạy lần đầu sau 1 phút, sau đó mỗi DRIFT_INTERVAL
    run_drift
    LAST_DRIFT=$NOW
  fi

  # Sleep nhỏ để không busy-loop, nhưng đủ để Ctrl+C phản hồi nhanh
  SLEEP_NEXT=30
  # Cap sleep nếu sắp tới deadline
  if [ "$REMAIN" -lt "$SLEEP_NEXT" ]; then SLEEP_NEXT=$REMAIN; fi
  sleep "$SLEEP_NEXT"
done

# ---------- Phase 5: Final summary ----------
build_summary() {
  phase "Phase 5/5 — Build evidence summary"
  local END_TS=$(date +%s)
  local TOTAL_ELAPSED=$(( END_TS - START_TS ))
  local N_SNAPSHOTS N_DRIFT N_TIMELINE
  N_SNAPSHOTS=$(find "$OUT_DIR/snapshots" -name '*_containers.txt' 2>/dev/null | wc -l | tr -d ' ')
  N_DRIFT=$(find "$OUT_DIR/drift" -name '*_drift.html' 2>/dev/null | wc -l | tr -d ' ')
  N_TIMELINE=$(( $(wc -l < "$TIMELINE_CSV" 2>/dev/null || echo 1) - 1 ))

  local SUMMARY="$OUT_DIR/EVIDENCE_SUMMARY.md"
  {
    echo "# Evidence Summary — $TIMESTAMP"
    echo
    echo "**Tổng thời lượng:** $(elapsed_human $TOTAL_ELAPSED) (dự kiến ${HOURS}h)"
    echo "**Bắt đầu:** $(date -r "$START_TS" +'%F %T')"
    echo "**Kết thúc:** $(date -r "$END_TS" +'%F %T')"
    echo
    echo "## Thông số chạy"
    echo
    echo "| Tham số | Giá trị |"
    echo "|---|---|"
    echo "| skip-train | $SKIP_TRAIN |"
    echo "| subset | $SUBSET_SIZE |"
    echo "| epochs | $EPOCHS |"
    echo "| imgsz | $IMGSZ |"
    echo "| batch | $BATCH |"
    echo "| snapshot interval | ${SNAPSHOT_INTERVAL}s |"
    echo "| traffic interval | ${TRAFFIC_INTERVAL}s (burst=$TRAFFIC_BURST) |"
    echo "| drift interval | ${DRIFT_INTERVAL}s |"
    echo
    echo "## Số liệu evidence"
    echo
    echo "| Mục | Số lượng |"
    echo "|---|---|"
    echo "| Snapshot Prometheus/MLflow/MinIO | $N_SNAPSHOTS |"
    echo "| Báo cáo drift HTML | $N_DRIFT |"
    echo "| Dòng timeline CSV | $N_TIMELINE |"
    echo
    echo "## Trạng thái cuối cùng"
    echo
    echo '```'
    docker ps --format '{{.Names}}|{{.Status}}' 2>/dev/null | sort || echo "(docker ps fail)"
    echo '```'
    echo
    echo "## Cấu trúc output"
    echo
    echo '```'
    (cd "$OUT_DIR" && find . -maxdepth 2 -type d 2>/dev/null | sort)
    echo '```'
    echo
    echo "## Cách dùng làm minh chứng cho báo cáo"
    echo
    echo "1. **Hình hệ thống vận hành liên tục**: mở \`timeline.csv\` trong Excel/pandas → vẽ chart \`http_requests_total\` và 4 cột \`*_up\` theo trục thời gian."
    echo "2. **Hình MLflow tích luỹ run**: chạy \`jq '.runs | length' snapshots/*_mlflow_runs.json\` → biểu đồ số run theo thời gian."
    echo "3. **Hình drift theo thời gian**: mở các file \`drift/*_drift.html\` lần lượt → screenshot Deepchecks dashboard."
    echo "4. **Snapshot Prometheus**: dùng cho phụ lục \"hệ thống monitoring giữ trạng thái healthy trong N giờ\"."
    echo
    echo "Tất cả file thô đều nằm trong \`$OUT_DIR\`."
  } > "$SUMMARY"

  ok "Summary tại: $SUMMARY"
  log "Snapshots:   $N_SNAPSHOTS"
  log "Drift HTML:  $N_DRIFT"
  log "Timeline:    $N_TIMELINE dòng → $TIMELINE_CSV"
}

build_summary
phase "Done"
log "Toàn bộ evidence ở: $OUT_DIR"
exit 0
