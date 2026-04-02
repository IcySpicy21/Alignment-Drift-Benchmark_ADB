#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SEEDS=(11 22 33)
PRECS=(fp16 int8 int4)
TEMP="${TEMP:-0.7}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"
MODEL="mistralai/Mistral-7B-Instruct-v0.2"

for prec in "${PRECS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running Mistral fast slice: precision=$prec seed=$seed temp=$TEMP max_prompts=$MAX_PROMPTS"
    "$PYTHON_BIN" evaluation/evaluate_alignment_drift.py \
      --models "$MODEL" \
      --precision "$prec" \
      --quant-method bitsandbytes \
      --seed "$seed" \
      --temperature "$TEMP" \
      --max-prompts "$MAX_PROMPTS"
  done
done

echo "Refreshing analysis outputs"
"$PYTHON_BIN" analysis/run_analysis.py

echo "Done. Check analysis/seed_aggregate_*.csv and figures/*.pdf"
