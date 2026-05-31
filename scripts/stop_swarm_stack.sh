#!/usr/bin/env bash
#
# Dừng Swarm stack mlops và (tuỳ chọn) leave swarm.
#
# Usage:
#   bash scripts/stop_swarm_stack.sh                  # chỉ rm stack
#   bash scripts/stop_swarm_stack.sh --leave-swarm    # rm stack + leave swarm (xoá toàn bộ overlay)
#   bash scripts/stop_swarm_stack.sh --keep-registry  # giữ local registry
#
set -uo pipefail

STACK_NAME="mlops"
LEAVE_SWARM=0
KEEP_REGISTRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --leave-swarm)    LEAVE_SWARM=1; shift ;;
    --keep-registry)  KEEP_REGISTRY=1; shift ;;
    -h|--help)
      awk '/^[^#]/{exit} /^#!/{next} /^#/{sub(/^# ?/,""); print}' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Removing stack $STACK_NAME"
docker stack rm "$STACK_NAME" 2>&1 | tail -5 || true

log "Đợi cleanup task (10s)"
sleep 10

if [ "$KEEP_REGISTRY" -eq 0 ]; then
  if docker service inspect registry >/dev/null 2>&1; then
    log "Removing local registry"
    docker service rm registry >/dev/null
  fi
fi

if [ "$LEAVE_SWARM" -eq 1 ]; then
  log "Leave swarm (force)"
  docker swarm leave --force 2>&1 | tail -3
fi

log "Done. Trạng thái:"
docker stack ls 2>&1 | head -5
docker service ls 2>&1 | head -10
