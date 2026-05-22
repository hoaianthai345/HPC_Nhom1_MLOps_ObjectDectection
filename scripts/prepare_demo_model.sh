#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/prepare_demo_model.sh /path/to/model.pt"
  exit 1
fi

MODEL_SOURCE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$MODEL_SOURCE" ]; then
  echo "Model file not found: $MODEL_SOURCE"
  exit 1
fi

mkdir -p "$REPO_ROOT/model_artifacts"
cp "$MODEL_SOURCE" "$REPO_ROOT/serving_model.pt"

case "$(basename "$MODEL_SOURCE")" in
  *teacher*)
    cp "$MODEL_SOURCE" "$REPO_ROOT/model_artifacts/teacher_best.pt"
    ;;
  *student_kd*|*kd*)
    cp "$MODEL_SOURCE" "$REPO_ROOT/model_artifacts/student_kd_best.pt"
    ;;
  *student*)
    cp "$MODEL_SOURCE" "$REPO_ROOT/model_artifacts/student_best.pt"
    ;;
esac

echo "Prepared serving model:"
ls -lh "$REPO_ROOT/serving_model.pt"
