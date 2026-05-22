#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/dist"
STAMP="$(date +%Y%m%d_%H%M%S)"
PACKAGE="$OUT_DIR/hpc_nhom1_artifacts_$STAMP.zip"

mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
zip -r "$PACKAGE" \
  README.md \
  PROJECT_EXECUTION.md \
  docs \
  assets \
  notebooks/colab_train.ipynb \
  notebooks/trainer.py \
  scripts \
  training_pipeline/src/config \
  reports \
  model_artifacts \
  -x "*.DS_Store" \
  -x "__pycache__/*" \
  -x "*.pyc" \
  -x ".venv-demo/*" \
  -x "dist/*"

echo "Created package:"
ls -lh "$PACKAGE"
