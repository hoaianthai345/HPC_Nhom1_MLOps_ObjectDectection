#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-demo"
LOG_DIR="$REPO_ROOT/reports/logs"
MODEL_PATH="${YOLO_MODEL_PATH:-$REPO_ROOT/serving_model.pt}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model file: $MODEL_PATH"
  echo "Run: bash scripts/prepare_demo_model.sh /path/to/model.pt"
  exit 1
fi

mkdir -p "$LOG_DIR"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$REPO_ROOT/serving_pipeline/requirements.demo.txt"

export PYTHONPATH="$REPO_ROOT"
export YOLO_MODEL_PATH="$MODEL_PATH"
export DEVICE="${DEVICE:-cpu}"
export USE_MINIO=false
export ENABLE_DRIFT_API=false
export HOST="${HOST:-127.0.0.1}"
export API_HOST=localhost
export API_PORT=8000
export GRADIO_PORT=7860
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp}"
export PRODUCTION_DIR="$REPO_ROOT/serving_pipeline/production"
export PRODUCTION_DATA_DIR="$REPO_ROOT/serving_pipeline/production-data"

if [ -f "$LOG_DIR/api.pid" ]; then
  old_pid="$(cat "$LOG_DIR/api.pid")"
  kill "$old_pid" 2>/dev/null || true
fi

if [ -f "$LOG_DIR/gradio.pid" ]; then
  old_pid="$(cat "$LOG_DIR/gradio.pid")"
  kill "$old_pid" 2>/dev/null || true
fi

cd "$REPO_ROOT"
nohup "$VENV_DIR/bin/uvicorn" serving_pipeline.api.main:app --host "$HOST" --port "$API_PORT" > "$LOG_DIR/api.log" 2>&1 &
echo "$!" > "$LOG_DIR/api.pid"

sleep 5

nohup "$VENV_DIR/bin/python" -m serving_pipeline.gradio_app > "$LOG_DIR/gradio.log" 2>&1 &
echo "$!" > "$LOG_DIR/gradio.pid"

echo "Local demo started."
echo "FastAPI: http://localhost:8000/docs"
echo "Gradio : http://localhost:7860"
echo "Logs   : $LOG_DIR/api.log"
echo "Logs   : $LOG_DIR/gradio.log"
