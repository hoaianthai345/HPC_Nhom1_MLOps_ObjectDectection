#!/usr/bin/env bash
#
# Warm-up traffic cho Grafana trước buổi demo.
#
# Bắn N request vào FastAPI /detect để Prometheus scrape có dữ liệu time-series,
# Grafana dashboard mới có panel để hiển thị thay vì trống trơn.
#
# Usage:
#   bash scripts/demo_warmup.sh                  # 60 request, mỗi request cách 1s
#   bash scripts/demo_warmup.sh -n 200           # 200 request
#   bash scripts/demo_warmup.sh -n 300 -i 0.5    # 300 request, cách 0.5s
#   bash scripts/demo_warmup.sh --image path/to/x.jpg  # ảnh tuỳ chỉnh
#   bash scripts/demo_warmup.sh --background     # chạy nền (return PID để kill sau)
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_REQUESTS=60
INTERVAL=1
IMAGE=""
BACKGROUND=0
API_URL="http://localhost:8000/detect"
CONF="0.25"
IOU="0.45"

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--num)        N_REQUESTS="$2"; shift 2 ;;
    -i|--interval)   INTERVAL="$2"; shift 2 ;;
    --image)         IMAGE="$2"; shift 2 ;;
    --url)           API_URL="$2"; shift 2 ;;
    --background)    BACKGROUND=1; shift ;;
    -h|--help)
      awk '/^[^#]/{exit} /^#!/{next} /^#/{sub(/^# ?/,""); print}' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

# Auto-pick sample image nếu không truyền
if [ -z "$IMAGE" ]; then
  for d in "$REPO_ROOT/data/demo_subset/valid/images" \
           "$REPO_ROOT/data/demo_subset/test/images" \
           "$REPO_ROOT/data/raw/test/images"; do
    if [ -d "$d" ]; then
      IMAGE=$(find "$d" -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | head -1)
      [ -n "$IMAGE" ] && break
    fi
  done
fi

[ -f "$IMAGE" ] || { echo "❌  Không tìm thấy ảnh demo. Truyền --image <path>" >&2; exit 1; }

# Verify FastAPI up
if ! curl -sf -o /dev/null http://localhost:8000/health 2>/dev/null; then
  echo "❌  FastAPI chưa lên tại http://localhost:8000/health" >&2
  echo "   Chạy: bash scripts/start_full_local.sh" >&2
  exit 1
fi

echo "[warmup] Image: $(basename "$IMAGE")"
echo "[warmup] Target: $API_URL"
echo "[warmup] Requests: $N_REQUESTS, interval: ${INTERVAL}s"
echo "[warmup] Tổng thời gian dự kiến: $(echo "$N_REQUESTS * $INTERVAL" | bc 2>/dev/null || echo "~$(( N_REQUESTS * 1 ))" )s"

run_traffic() {
  local ok=0 fail=0
  for i in $(seq 1 "$N_REQUESTS"); do
    if curl -sf -o /dev/null -F "file=@${IMAGE}" \
         "${API_URL}?confidence_threshold=${CONF}&iou_threshold=${IOU}"; then
      ok=$((ok+1))
    else
      fail=$((fail+1))
    fi
    # Progress mỗi 20 request
    if [ $((i % 20)) -eq 0 ]; then
      echo "[warmup] $i/$N_REQUESTS  (OK=$ok, fail=$fail)"
    fi
    sleep "$INTERVAL"
  done
  echo "[warmup] Hoàn tất: OK=$ok, fail=$fail"
  echo "[warmup] Mở Grafana: http://localhost:3000 (admin/admin)"
}

if [ "$BACKGROUND" -eq 1 ]; then
  ( run_traffic ) >> "$REPO_ROOT/reports/logs/demo_warmup.log" 2>&1 &
  PID=$!
  echo "[warmup] Chạy nền với PID=$PID"
  echo "[warmup] Để kill sau: kill $PID"
  echo "[warmup] Log: reports/logs/demo_warmup.log"
else
  run_traffic
fi
