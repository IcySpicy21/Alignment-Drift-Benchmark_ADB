#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODELS=(
  "google/gemma-2b-it"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "meta-llama/Meta-Llama-3-8B-Instruct"
)
PRECS=(fp16 int8 int4)
SEEDS=(11 22 33 44 55)
TEMP="${TEMP:-0.7}"
MAX_PROMPTS="${MAX_PROMPTS:-}"

for model in "${MODELS[@]}"; do
  for prec in "${PRECS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      CMD=(
        "$PYTHON_BIN" evaluation/evaluate_alignment_drift.py
        --models "$model"
        --precision "$prec"
        --quant-method bitsandbytes
        --seed "$seed"
        --temperature "$TEMP"
      )

      if [[ -n "$MAX_PROMPTS" ]]; then
        CMD+=(--max-prompts "$MAX_PROMPTS")
      fi

      echo "Running: model=$model precision=$prec seed=$seed temp=$TEMP"
      "${CMD[@]}"
    done
  done
done

echo "Matrix run complete. Next: $PYTHON_BIN analysis/run_analysis.py"
