# Venue-strength backlog (internal)

Checklist for tightening evidence; **do not** treat this markdown as citable text.

## P1 — RNG / variance

Run 3–5 seeds per `(model, precision)` and archive aggregates.

```bash
python evaluation/evaluate_alignment_drift.py \
  --models mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct google/gemma-2b-it \
  --precision fp16 int8 int4 \
  --seed 11 22 33 44 55 \
  --temperature 0.7
python analysis/run_analysis.py
```

Deliverables: `analysis/seed_aggregate_*.csv` (or successor filenames your branch emits), refreshed plots.

## P2 — More families

Add instruction-tuned checkpoints outside the original trio (Qwen, Phi, MoE class, etc.) using the same harness.

## P3 — Human adjudication

Two annotators, merge script, confusion exports; regenerate `paper/judge_validation_paragraph.tex` from **human** merges only.

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

## P4 — Quantizer sensitivity

Match prompts/seeds across ≥2 INT4 stacks:

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

## P5 — Utility vs refusal chart

Regenerate when capability CSV changes:

```bash
python analysis/plot_safety_utility_tradeoff.py
```

Inputs: `analysis/refusal_summary.csv`, `analysis/capability_control.csv`  
Output: `figures/safety_utility_tradeoff.pdf`

## Pre-submit sanity (venue mode)

- Cross-seed stability referenced **only** if CSV proves it.  
- Human-judge metrics + disagreement slices present.  
- Multi-family table present (not one row).  
- Quantizer section present.  
- Utility/refusal figure discussed with operational interpretation.
