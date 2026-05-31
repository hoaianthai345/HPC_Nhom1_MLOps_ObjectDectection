#!/usr/bin/env bash
#
# Khởi động Docker Swarm stack: serving (api + ui) + monitor (prom/grafana/...)
#
# Quy trình:
#   1) Init swarm (nếu chưa)
#   2) Tạo overlay network mlops-overlay (attachable để compose stack ngoài kết nối)
#   3) Bật local registry tại :5000
#   4) Build + tag + push 2 image serving
#   5) Deploy 2 stack file
#   6) Đợi converge + in trạng thái
#
# Usage:
#   bash scripts/start_swarm_stack.sh                 # full flow
#   bash scripts/start_swarm_stack.sh --no-build      # bỏ qua rebuild image
#   bash scripts/start_swarm_stack.sh --serving-only  # chỉ deploy serving stack
#   bash scripts/start_swarm_stack.sh --help
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK_NAME="mlops"
REGISTRY="localhost:5000"
NETWORK="mlops-overlay"
NO_BUILD=0
SERVING_ONLY=0
MONITOR_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build)      NO_BUILD=1; shift ;;
    --serving-only)  SERVING_ONLY=1; shift ;;
    --monitor-only)  MONITOR_ONLY=1; shift ;;
    -h|--help)
      awk '/^[^#]/{exit} /^#!/{next} /^#/{sub(/^# ?/,""); print}' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "❌  $*"; exit 1; }
ok()   { log "✅  $*"; }
step() { echo; log "── $* ──"; }

# 1) Init swarm
step "1/6  Init swarm"
state=$(docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null || echo "error")
if [ "$state" != "active" ]; then
  docker swarm init --advertise-addr 127.0.0.1 >/dev/null || fail "docker swarm init failed"
  ok "Swarm initialized"
else
  ok "Swarm đã active"
fi

# 2) Overlay network
step "2/6  Tạo overlay network $NETWORK"
if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
  docker network create --driver overlay --attachable "$NETWORK" >/dev/null \
    || fail "Không tạo được overlay network"
  ok "Network $NETWORK created"
else
  ok "Network $NETWORK đã có"
fi

# 3) Local registry
step "3/6  Local registry tại :5000"
if ! docker service inspect registry >/dev/null 2>&1; then
  docker service create --name registry --publish 5000:5000 \
    --restart-condition any registry:2 >/dev/null \
    || fail "Không start được local registry"
  ok "Registry created — chờ 5s..."
  sleep 5
else
  ok "Registry đã có"
fi

# 4) Build + push image
if [ "$NO_BUILD" -eq 0 ] && [ "$MONITOR_ONLY" -eq 0 ]; then
  step "4/6  Build + push image"
  (cd "$REPO_ROOT" && \
   docker build -f serving_pipeline/docker/Dockerfile.backend \
     -t "$REGISTRY/serving-api:v1" .) || fail "Build api thất bại"
  docker push "$REGISTRY/serving-api:v1" >/dev/null || fail "Push api thất bại"
  ok "serving-api:v1 đã push"

  (cd "$REPO_ROOT" && \
   docker build -f serving_pipeline/docker/Dockerfile.ui \
     -t "$REGISTRY/serving-ui:v1" .) || fail "Build ui thất bại"
  docker push "$REGISTRY/serving-ui:v1" >/dev/null || fail "Push ui thất bại"
  ok "serving-ui:v1 đã push"
else
  log "Bỏ qua build/push (--no-build hoặc --monitor-only)"
fi

# 5) Deploy stack
step "5/6  Deploy stack"
if [ "$MONITOR_ONLY" -eq 0 ]; then
  docker stack deploy -c "$REPO_ROOT/infra/swarm/docker-stack.serving.yml" "$STACK_NAME" \
    || fail "Deploy serving stack thất bại"
  ok "Serving stack deployed"
fi
if [ "$SERVING_ONLY" -eq 0 ]; then
  (cd "$REPO_ROOT/infra/swarm" && \
   docker stack deploy -c docker-stack.monitor.yml "$STACK_NAME") \
    || fail "Deploy monitor stack thất bại"
  ok "Monitor stack deployed"
fi

# 6) Đợi converge
step "6/6  Đợi service converge (timeout 90s)"
DEADLINE=$(($(date +%s) + 90))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  pending=$(docker stack services "$STACK_NAME" --format '{{.Replicas}}' \
            | awk -F/ '$1 != $2 {n++} END {print n+0}')
  if [ "$pending" -eq 0 ]; then break; fi
  sleep 3
done

echo
log "── TRẠNG THÁI STACK $STACK_NAME ──"
docker stack services "$STACK_NAME"
echo
log "── TASKS ──"
docker stack ps "$STACK_NAME" --filter "desired-state=running" --format \
  'table {{.Name}}\t{{.Node}}\t{{.CurrentState}}\t{{.Ports}}'
echo
log "Truy cập:"
log "  FastAPI:     http://localhost:8000/docs"
log "  Gradio UI:   http://localhost:7860"
log "  Prometheus:  http://localhost:9090"
log "  Grafana:     http://localhost:3000  (admin/admin)"
log "  Alertmanager: http://localhost:9093"
