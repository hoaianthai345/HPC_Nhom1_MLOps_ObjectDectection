#!/usr/bin/env bash
#
# Bring up the full MLOps stack on a single local machine for demo/báo cáo.
#
# Default stack:  MLflow + MinIO + Monitor (Prometheus/Grafana/Loki) + Serving
# With Airflow:   add `--with-airflow` (heavier, ~4 more containers)
# Minimal demo:   `--minimal`  → only Serving (FastAPI + Gradio)
#
# Usage:
#   bash scripts/start_full_local.sh                 # default full stack
#   bash scripts/start_full_local.sh --with-airflow  # + Airflow
#   bash scripts/start_full_local.sh --minimal       # only API + UI
#   bash scripts/start_full_local.sh --no-build      # skip docker image rebuild
#   MLFLOW_HOST_PORT=5001 bash scripts/start_full_local.sh  # force host ports when needed
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWORK_NAME="hpc-nhom1-network"

WITH_AIRFLOW=0
MINIMAL=0
NO_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --with-airflow) WITH_AIRFLOW=1 ;;
    --minimal)      MINIMAL=1 ;;
    --no-build)     NO_BUILD=1 ;;
    -h|--help)
      awk '/^[^#]/{exit} /^#!/{next} /^#/{sub(/^# ?/,""); print}' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

build_flag=""
[ "$NO_BUILD" -eq 0 ] && build_flag="--build"
RESERVED_HOST_PORTS=""

# ---------- helpers ----------
log()   { printf '\033[1;34m[start]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m  ok   \033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m  warn \033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m  fail \033[0m %s\n' "$*"; }

ensure_env() {
  local dir="$1"
  if [ -f "$dir/.env.example" ] && [ ! -f "$dir/.env" ]; then
    cp "$dir/.env.example" "$dir/.env"
    ok "Created $dir/.env from .env.example"
  fi
}

port_is_listening() {
  nc -z 127.0.0.1 "$1" >/dev/null 2>&1
}

container_publishes_port() {
  local container="$1" container_port="$2" host_port="$3"
  docker port "$container" "${container_port}/tcp" 2>/dev/null | grep -Eq ":${host_port}$"
}

choose_host_port() {
  # choose_host_port <environment_variable> <preferred_port> <container> <container_port>
  local var_name="$1" preferred="$2" container="$3" container_port="$4"
  local requested="${!var_name-}" chosen

  if [ -n "$requested" ]; then
    export "${var_name}=${requested}"
    RESERVED_HOST_PORTS="${RESERVED_HOST_PORTS} ${requested}"
    return
  fi

  chosen="$preferred"
  while { port_is_listening "$chosen" && ! container_publishes_port "$container" "$container_port" "$chosen"; } ||
        [[ " ${RESERVED_HOST_PORTS} " == *" ${chosen} "* ]]; do
    chosen=$((chosen + 1))
  done
  if [ "$chosen" != "$preferred" ]; then
    warn "Port $preferred is already in use; using $chosen for $container instead"
  fi
  export "${var_name}=${chosen}"
  RESERVED_HOST_PORTS="${RESERVED_HOST_PORTS} ${chosen}"
}

wait_http() {
  # wait_http <url> <name> <timeout_seconds>
  local url="$1" name="$2" timeout="${3:-60}"
  local start; start=$(date +%s)
  while true; do
    if curl -fsS -o /dev/null --max-time 3 "$url" 2>/dev/null; then
      ok "$name reachable at $url"
      return 0
    fi
    if [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
      fail "$name not reachable at $url within ${timeout}s"
      return 1
    fi
    sleep 2
  done
}

# ---------- pre-flight ----------
log "Pre-flight checks"
if ! command -v docker >/dev/null 2>&1; then
  fail "docker not installed"; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  fail "docker daemon not running"; exit 1
fi
ok "docker reachable"

if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  docker network create "$NETWORK_NAME" >/dev/null
  ok "Created docker network: $NETWORK_NAME"
else
  ok "Docker network $NETWORK_NAME already exists"
fi

if [ ! -f "$REPO_ROOT/serving_model.pt" ]; then
  warn "serving_model.pt missing — API sẽ tải yolo11n.pt mặc định lúc startup"
  warn "Nếu muốn dùng model đã train: bash scripts/prepare_demo_model.sh /path/to/model.pt"
fi

# ---------- bring up stacks ----------
if [ "$MINIMAL" -eq 0 ]; then
  choose_host_port MLFLOW_HOST_PORT 5000 mlflow_server 5000
  choose_host_port MINIO_API_HOST_PORT 9000 hpc_nhom1_minio 9000
  choose_host_port MINIO_CONSOLE_HOST_PORT 9001 hpc_nhom1_minio 9001

  log "Starting infra/mlflow (MLflow + MinIO + MySQL)"
  ensure_env "$REPO_ROOT/infra/mlflow"
  (cd "$REPO_ROOT/infra/mlflow" && docker compose up -d)
  wait_http "http://localhost:${MLFLOW_HOST_PORT}/" "MLflow" 90 || true
  wait_http "http://localhost:${MINIO_CONSOLE_HOST_PORT}/" "MinIO Console" 60 || true

  log "Starting infra/monitor (Prometheus + Grafana + Loki + Alertmanager)"
  (cd "$REPO_ROOT/infra/monitor" && docker compose up -d)
  wait_http "http://localhost:9090/-/healthy" "Prometheus" 60 || true
  wait_http "http://localhost:3000/api/health" "Grafana" 60 || true
  wait_http "http://localhost:3100/ready"      "Loki" 60 || true

  if [ "$WITH_AIRFLOW" -eq 1 ]; then
    log "Starting infra/airflow"
    ensure_env "$REPO_ROOT/infra/airflow"
    (cd "$REPO_ROOT/infra/airflow" && docker compose up -d)
    wait_http "http://localhost:8080/health" "Airflow" 180 || true
  fi
fi

log "Starting serving_pipeline (FastAPI + Gradio)"
(cd "$REPO_ROOT/serving_pipeline" && docker compose up -d $build_flag api ui)
wait_http "http://localhost:8000/health" "FastAPI" 120 || true
wait_http "http://localhost:7860/"       "Gradio"  90  || true

# ---------- summary ----------
echo
log "Demo stack is up. URLs:"
cat <<EOF

  Gradio UI       :  http://localhost:7860
  FastAPI docs    :  http://localhost:8000/docs
  FastAPI /metrics:  http://localhost:8000/metrics
EOF
if [ "$MINIMAL" -eq 0 ]; then
  cat <<EOF
  MLflow UI       :  http://localhost:${MLFLOW_HOST_PORT}
  MinIO API       :  http://localhost:${MINIO_API_HOST_PORT}
  MinIO Console   :  http://localhost:${MINIO_CONSOLE_HOST_PORT}   (admin: minio_admin / minio_password123)
  Prometheus      :  http://localhost:9090
  Grafana         :  http://localhost:3000   (admin / admin trừ khi đổi .env)
  Loki            :  http://localhost:3100/ready
  Alertmanager    :  http://localhost:9093
EOF
fi
if [ "$WITH_AIRFLOW" -eq 1 ]; then
  echo "  Airflow         :  http://localhost:8080"
fi

cat <<EOF

Healthcheck nhanh:    MLFLOW_HOST_PORT=${MLFLOW_HOST_PORT:-5000} MINIO_CONSOLE_HOST_PORT=${MINIO_CONSOLE_HOST_PORT:-9001} bash scripts/smoke_test_local.sh
Tắt toàn bộ stack:    bash scripts/stop_full_local.sh
Tắt + xoá volumes:    bash scripts/stop_full_local.sh --purge
EOF
