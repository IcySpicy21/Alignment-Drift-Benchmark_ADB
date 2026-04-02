#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="adb-gptq:latest"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

echo "[1/2] Building GPTQ Docker image: $IMAGE_NAME"
docker build -f docker/Dockerfile.gptq -t "$IMAGE_NAME" .

echo "[2/2] Running command inside container"

ENV_ARGS=()
if [[ -f ".env" ]]; then
  ENV_ARGS+=(--env-file .env)
fi

DEFAULT_CMD="python evaluation/evaluate_alignment_drift.py --models TheBloke/Mistral-7B-Instruct-v0.2-GPTQ --precision int4 --quant-method gptq --seed 21 --temperature 0.7 --max-prompts 1"
USER_CMD="${*:-$DEFAULT_CMD}"

docker run --rm --gpus all \
  "${ENV_ARGS[@]}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$IMAGE_NAME" \
  -lc "$USER_CMD"

echo "Done. Check evaluation/logs for generated CSV files."
