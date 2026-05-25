#!/usr/bin/env bash
#
# Smoke test the local MLOps stack — chạy 5 phút trước buổi demo để chắc chắn
# không sập tại chỗ.
#
# Checks:
#   FastAPI       /health
#   FastAPI       POST /detect với 1 ảnh mẫu (skip nếu không tìm thấy ảnh)
#   FastAPI       /metrics (Prometheus exposition)
#   Gradio        root
#   MLflow        UI root
#   MinIO         console
#   Prometheus    /-/healthy
#   Grafana       /api/health
#   Loki          /ready
#   Alertmanager  /-/healthy
#   Airflow       /health  (chỉ nếu container đang chạy)
#
# Exit code:
#   0  → mọi check bắt buộc đều pass
#   1  → có check bắt buộc fail
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT="${TIMEOUT:-4}"
DETECT_TIMEOUT="${DETECT_TIMEOUT:-120}"
MLFLOW_HOST_PORT="${MLFLOW_HOST_PORT:-5000}"
MINIO_CONSOLE_HOST_PORT="${MINIO_CONSOLE_HOST_PORT:-9001}"

PASS=0
FAIL=0
declare -a FAIL_NAMES=()

ok()   { printf '  \033[1;32m✓\033[0m  %s\n' "$*"; PASS=$((PASS+1)); }
ng()   { printf '  \033[1;31m✗\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); FAIL_NAMES+=("$1"); }
skip() { printf '  \033[1;33m·\033[0m  %s (skipped)\n' "$*"; }
hdr()  { printf '\n\033[1;34m== %s ==\033[0m\n' "$*"; }

check_http() {
  # check_http <name> <url> [expected_substring]
  local name="$1" url="$2" expect="${3:-}"
  local body
  body="$(curl -fsS --max-time "$TIMEOUT" "$url" 2>/dev/null)" || { ng "$name  ($url)"; return; }
  if [ -n "$expect" ] && ! grep -q "$expect" <<<"$body"; then
    ng "$name  ($url, expected '$expect')"; return
  fi
  ok "$name  ($url)"
}

# ---------- Serving ----------
hdr "Serving"
check_http "FastAPI /health"   "http://localhost:8000/health"
check_http "FastAPI /metrics"  "http://localhost:8000/metrics" "process_"

sample_img="$(find "$REPO_ROOT/serving_pipeline/production/images" -maxdepth 1 -type f -iname '*.jpg' -print -quit 2>/dev/null)"
if [ -z "$sample_img" ]; then
  sample_img="$(find "$REPO_ROOT/../sample_data" -maxdepth 1 -type f -iname '*.jpg' -print -quit 2>/dev/null)"
fi

if [ -n "$sample_img" ]; then
  resp="$(curl -fsS --max-time "$DETECT_TIMEOUT" -X POST \
    -F "file=@${sample_img}" \
    http://localhost:8000/detect 2>/dev/null || true)"
  if grep -q '"num_detections"' <<<"$resp"; then
    ndet="$(grep -oE '"num_detections":[0-9]+' <<<"$resp" | head -1 | cut -d: -f2)"
    ms="$(grep -oE '"inference_time_ms":[0-9.]+' <<<"$resp" | head -1 | cut -d: -f2)"
    ok "FastAPI POST /detect  ($(basename "$sample_img"): ${ndet} detections, ${ms}ms)"
  else
    ng "FastAPI POST /detect  (response did not contain num_detections)"
  fi
else
  skip "FastAPI POST /detect (không tìm thấy ảnh mẫu local)"
fi

check_http "Gradio UI"         "http://localhost:7860/"

# ---------- MLflow + MinIO ----------
hdr "MLflow + MinIO"
check_http "MLflow UI"         "http://localhost:${MLFLOW_HOST_PORT}/"
check_http "MinIO Console"     "http://localhost:${MINIO_CONSOLE_HOST_PORT}/"

# ---------- Monitoring ----------
hdr "Monitoring"
check_http "Prometheus"        "http://localhost:9090/-/healthy"
check_http "Grafana"           "http://localhost:3000/api/health" '"database"[[:space:]]*:[[:space:]]*"ok"'
check_http "Loki"              "http://localhost:3100/ready"
check_http "Alertmanager"      "http://localhost:9093/-/healthy"

# ---------- Airflow (only if container exists) ----------
hdr "Airflow"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "airflow"; then
  check_http "Airflow webserver" "http://localhost:8080/health" '"metadatabase"'
else
  skip "Airflow webserver (start_full_local.sh không bật, dùng --with-airflow)"
fi

# ---------- summary ----------
hdr "Summary"
printf "  Passed: \033[1;32m%d\033[0m   Failed: \033[1;31m%d\033[0m\n" "$PASS" "$FAIL"

if [ "$FAIL" -gt 0 ]; then
  printf "\n  Failing checks: %s\n" "${FAIL_NAMES[*]}"
  printf "  Xem log:        docker compose logs --tail=50 -f <service>\n"
  exit 1
fi
exit 0
