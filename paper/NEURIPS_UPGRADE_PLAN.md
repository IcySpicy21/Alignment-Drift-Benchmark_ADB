# NeurIPS-Oriented Upgrade Plan (V2)

This plan converts the current exploratory paper into a confirmatory package.

## Priority 1: Multi-seed robustness

Goal: run 3-5 seeds for each model x precision cell and report mean/std/CI.

Execution:

```bash
python evaluation/evaluate_alignment_drift.py \
  --models mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct google/gemma-2b-it \
  --precision fp16 int8 int4 \
  --seed 11 22 33 44 55 \
  --temperature 0.7
python analysis/run_analysis.py
```

Deliverables:
- seed aggregates in analysis/seed_aggregate_*.csv
- updated figures and summaries

## Priority 2: Expand model coverage

Goal: reduce model-family artifact risk.

Recommended additions:
- Qwen (instruction-tuned)
- Phi (small model family)
- Mixtral or another MoE model
- Falcon-style baseline

Deliverables:
- same precision/seed matrix and summary tables for added families

## Priority 3: Human judge validation (real, not proxy)

Goal: validate judge reliability with two independent annotators.

Workflow:

```bash
python analysis/prepare_human_annotation_packet.py \
  --input analysis/judge_annotations.csv \
  --outdir analysis/annotation_packet \
  --n 200 \
  --seed 42
# collect annotator_a.csv and annotator_b.csv
python analysis/merge_human_annotations.py
python analysis/run_judge_validation_refresh.py \
  --input analysis/judge_annotations_human_merged.csv \
  --label-source human
```

Deliverables:
- agreement summary and confusion matrix
- human-vs-judge paragraph in paper/judge_validation_paragraph.tex

## Priority 4: Quantization method robustness

Goal: show result is not specific to one quantizer.

Methods to add:
- AWQ
- GPTQ
- SmoothQuant (if architecture supports)

Evaluator support is now available via:

```bash
python evaluation/evaluate_alignment_drift.py \
  --models <repo_or_checkpoint> \
  --precision int4 \
  --quant-method awq \
  --seed 11 \
  --temperature 0.7

python evaluation/evaluate_alignment_drift.py \
  --models <repo_or_checkpoint> \
  --precision int4 \
  --quant-method gptq \
  --seed 11 \
  --temperature 0.7
```

Deliverables:
- method-wise drift table with matched prompts/seeds

## Priority 5: Safety-utility tradeoff figure

Goal: show refusal changes relative to utility change in one interpretable figure.

Implemented tooling:

```bash
python analysis/plot_safety_utility_tradeoff.py
```

Inputs:
- analysis/refusal_summary.csv
- analysis/capability_control.csv

Output:
- figures/safety_utility_tradeoff.pdf

## Acceptance-focused checks before V2 submission

- At least one key effect remains significant after correction across seeds
- Human-vs-judge metrics reported with confusion matrix and disagreement slices
- Multi-family consistency demonstrated (not only one model)
- Quantization method sensitivity reported
- Safety-utility tradeoff discussed with practical thresholds
