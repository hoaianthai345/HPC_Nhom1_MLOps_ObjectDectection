#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/reports/logs"

for name in api gradio; do
  pid_file="$LOG_DIR/$name.pid"
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill "$pid" 2>/dev/null; then
      echo "Stopped $name process: $pid"
    else
      echo "$name process was not running: $pid"
    fi
    rm -f "$pid_file"
  else
    echo "No PID file for $name"
  fi
done
