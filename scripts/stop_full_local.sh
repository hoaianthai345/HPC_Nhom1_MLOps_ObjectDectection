#!/usr/bin/env bash
#
# Tear down the full local MLOps stack started by `start_full_local.sh`.
#
# Usage:
#   bash scripts/stop_full_local.sh                 # stop containers, keep volumes + network
#   bash scripts/stop_full_local.sh --purge         # also remove docker volumes (mất MLflow runs, MinIO data, Grafana dashboards)
#   bash scripts/stop_full_local.sh --remove-network  # also remove the shared docker network
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWORK_NAME="hpc-nhom1-network"

PURGE=0
REMOVE_NETWORK=0
for arg in "$@"; do
  case "$arg" in
    --purge)          PURGE=1 ;;
    --remove-network) REMOVE_NETWORK=1 ;;
    -h|--help)
      awk '/^[^#]/{exit} /^#!/{next} /^#/{sub(/^# ?/,""); print}' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[stop]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ok  \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn \033[0m %s\n' "$*"; }

down_args="down"
[ "$PURGE" -eq 1 ] && down_args="down -v"

# Reverse order of start to respect dependencies.
for stack in serving_pipeline infra/airflow infra/monitor infra/mlflow; do
  if [ -f "$REPO_ROOT/$stack/docker-compose.yml" ]; then
    log "Bringing down $stack"
    if (cd "$REPO_ROOT/$stack" && docker compose $down_args --remove-orphans 2>/dev/null); then
      ok "$stack stopped"
    else
      warn "$stack compose down returned non-zero (likely was not running)"
    fi
  fi
done

if [ "$REMOVE_NETWORK" -eq 1 ]; then
  if docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    log "Removing docker network $NETWORK_NAME"
    docker network rm "$NETWORK_NAME" >/dev/null && ok "network removed"
  fi
fi

if [ "$PURGE" -eq 1 ]; then
  warn "Volumes removed. Lần chạy tiếp theo sẽ bắt đầu từ database trống."
fi

ok "Done."
