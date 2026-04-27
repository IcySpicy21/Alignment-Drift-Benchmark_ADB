# V2 GPU Command Sheet

Shell snippets only (no thesis prose). Run everything from the repo root unless noted.

## 0) Environment setup

```bash
source .venv/bin/activate  # or your Python env
python -m pip install -r requirements.txt
export HF_TOKEN=hf_your_token_here
export PYTHONUNBUFFERED=1
```

Notes:

- For gated models (for example `google/gemma-2b-it`), request/accept access on Hugging Face first.
- If you prefer CLI auth, run `huggingface-cli login` once in this environment.
- Optional method backends for INT4 comparisons:
  - `python -m pip install autoawq`
  - `python -m pip install auto-gptq`

## 1) Single-run sanity check (one model, one precision)

```bash
python evaluation/evaluate_alignment_drift.py \
  --models mistralai/Mistral-7B-Instruct-v0.2 \
  --precision fp16 \
  --seed 11 \
  --temperature 0.7
```

## 2) Baseline matrix (3 models x 3 precisions x seed=11, temp=0.7)

```bash
MODELS=(
  google/gemma-2b-it
  mistralai/Mistral-7B-Instruct-v0.2
  meta-llama/Meta-Llama-3-8B-Instruct
)
PRECS=(fp16 int8 int4)

for m in "${MODELS[@]}"; do
  for p in "${PRECS[@]}"; do
    python evaluation/evaluate_alignment_drift.py \
      --models "$m" \
      --precision "$p" \
      --seed 11 \
      --temperature 0.7
  done
done
```

## 3) Multi-seed expansion (5 seeds x temp=0.7)

```bash
MODELS=(
  google/gemma-2b-it
  mistralai/Mistral-7B-Instruct-v0.2
  meta-llama/Meta-Llama-3-8B-Instruct
)
PRECS=(fp16 int8 int4)
SEEDS=(11 22 33 44 55)

for m in "${MODELS[@]}"; do
  for p in "${PRECS[@]}"; do
    for s in "${SEEDS[@]}"; do
      python evaluation/evaluate_alignment_drift.py \
        --models "$m" \
        --precision "$p" \
        --seed "$s" \
        --temperature 0.7
    done
  done
done
```

## 4) Temperature sensitivity (T=0.0 vs T=0.7)

```bash
MODELS=(
  google/gemma-2b-it
  mistralai/Mistral-7B-Instruct-v0.2
  meta-llama/Meta-Llama-3-8B-Instruct
)
PRECS=(fp16 int8 int4)
TEMPS=(0.0 0.7)

for m in "${MODELS[@]}"; do
  for p in "${PRECS[@]}"; do
    for t in "${TEMPS[@]}"; do
      python evaluation/evaluate_alignment_drift.py \
        --models "$m" \
        --precision "$p" \
        --seed 11 \
        --temperature "$t"
    done
  done
done
```

## 5) Optional multi-GPU parallel launcher (4 workers)

Each worker pins to one GPU using `CUDA_VISIBLE_DEVICES`. Adjust `GPU_IDS` to your machine.

```bash
GPU_IDS=(0 1 2 3)
MODELS=(
  google/gemma-2b-it
  mistralai/Mistral-7B-Instruct-v0.2
  meta-llama/Meta-Llama-3-8B-Instruct
)
PRECS=(fp16 int8 int4)
SEEDS=(11 22 33 44 55)

jobs=()
for m in "${MODELS[@]}"; do
  for p in "${PRECS[@]}"; do
    for s in "${SEEDS[@]}"; do
      jobs+=("$m|$p|$s")
    done
  done
done

for i in "${!jobs[@]}"; do
  gpu="${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}"
  IFS='|' read -r m p s <<< "${jobs[$i]}"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python evaluation/evaluate_alignment_drift.py \
      --models "$m" \
      --precision "$p" \
      --seed "$s" \
      --temperature 0.7
  ) &

done

wait
```

## 6) Regenerate analysis outputs

```bash
python analysis/run_analysis.py
```

## 6b) Quantization-method robustness (INT4 backend comparisons)

Use the same model with different INT4 backends where quantized checkpoints are available.

```bash
python evaluation/evaluate_alignment_drift.py \
  --models mistralai/Mistral-7B-Instruct-v0.2 \
  --precision int4 \
  --quant-method bitsandbytes \
  --seed 11 \
  --temperature 0.7

python evaluation/evaluate_alignment_drift.py \
  --models <awq_or_gptq_model_repo> \
  --precision int4 \
  --quant-method awq \
  --seed 11 \
  --temperature 0.7

python evaluation/evaluate_alignment_drift.py \
  --models <awq_or_gptq_model_repo> \
  --precision int4 \
  --quant-method gptq \
  --seed 11 \
  --temperature 0.7
```

Fast smoke-test mode (per-category prompt cap):

```bash
python evaluation/evaluate_alignment_drift.py \
  --models mistralai/Mistral-7B-Instruct-v0.2 \
  --precision int4 \
  --quant-method bitsandbytes \
  --seed 13 \
  --temperature 0.7 \
  --max-prompts 1
```

Current environment notes:

- AWQ path is validated with `TheBloke/Mistral-7B-Instruct-v0.2-AWQ`.
- GPTQ path requires `auto-gptq`; installation may fail on Python 3.12 in some CUDA stacks.

## 6c) GPTQ fallback via Docker (Python 3.10 stack)

When host Python/CUDA fails for `auto-gptq`, use the dedicated container flow:

```bash
bash scripts/run_gptq_docker.sh
```

Run a custom command in the GPTQ container:

```bash
bash scripts/run_gptq_docker.sh \
  "python evaluation/evaluate_alignment_drift.py --models TheBloke/Mistral-7B-Instruct-v0.2-GPTQ --precision int4 --quant-method gptq --seed 21 --temperature 0.7 --max-prompts 25"
```

The helper script mounts the repository, loads `.env` when present, and writes logs to `evaluation/logs/` in your host workspace.

## 7) Resume behavior

The evaluator skips a run if the exact output CSV already exists.

- Baseline naming remains unchanged for `seed=42` and `temperature=0.7`.
- Non-baseline runs use filenames with `_seed<k>_temp<x>` suffixes to avoid collisions.

## 8) Judge validation from human annotations

First prepare a stratified annotation batch from evaluation logs:

```bash
python analysis/prepare_judge_annotation_batch.py \
  --glob 'evaluation/logs/results_*_seed*_temp*.csv' \
  --n 200 \
  --seed 42 \
  --out analysis/judge_annotations.csv
```

Then fill the `human_label` column in `analysis/judge_annotations.csv` (keep `judge_label` unchanged).

Optional: if you need a tiny manual starter instead, use the template file:

```bash
cp analysis/judge_annotation_template.csv analysis/judge_annotations.csv
# Fill rows in analysis/judge_annotations.csv with real annotations.
```

Then compute agreement artifacts:

```bash
python analysis/compute_judge_agreement.py \
  --input analysis/judge_annotations.csv \
  --outdir analysis

python analysis/render_judge_validation_paragraph.py \
  --summary analysis/judge_agreement_summary.csv \
  --category analysis/judge_category_disagreement.csv \
  --output paper/judge_validation_paragraph.tex
```

This writes:

- `analysis/judge_agreement_summary.csv`
- `analysis/judge_confusion_matrix.csv`
- `analysis/judge_category_disagreement.csv`

One-command refresh (after annotation):

```bash
python analysis/run_judge_validation_refresh.py \
  --input analysis/judge_annotations.csv
```
