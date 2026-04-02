# Phase 2 Execution Checklist (Exploratory -> Confirmatory)

This checklist operationalizes the current critique into concrete upgrade tasks.

## Current Position

- arXiv readiness: yes (exploratory framing is appropriate)
- conference readiness: not yet confirmatory
- primary bottlenecks: seed robustness, model breadth, human validation, method robustness, mechanism depth

## Workstream A: Multi-seed robustness (highest priority)

Status: in progress

Target:
- at least 5 seeds per model x precision cell
- report mean, std, bootstrap CI

Commands:

```bash
bash scripts/run_v2_matrix.sh
python analysis/run_analysis.py
```

Acceptance checks:
- seed aggregate files populated for core cells
- at least one key comparison remains stable across seeds

## Workstream B: Model-family expansion

Status: pending

Target additions:
- Qwen
- Phi
- Mixtral (or equivalent MoE)

Acceptance checks:
- added families included in refusal, drift, and margin summaries
- directional effect assessed across families, not one model only

## Workstream C: Human-vs-judge validation

Status: tooling complete, true annotation pending

Target:
- 200+ rows, 2 annotators, adjudication

Commands:

```bash
python analysis/prepare_human_annotation_packet.py --input analysis/judge_annotations.csv --outdir analysis/annotation_packet --n 200 --seed 42
# collect annotator_a.csv and annotator_b.csv
python analysis/merge_human_annotations.py
python analysis/run_judge_validation_refresh.py --input analysis/judge_annotations_human_merged.csv --label-source human
```

Acceptance checks:
- paragraph source is human-vs-judge (not proxy-vs-judge)
- disagreement slices and confusion matrix are included

## Workstream D: Quantization-method robustness

Status: partial (bitsandbytes and AWQ available, GPTQ blocked on host stack)

Target:
- compare INT4 backends for matched prompts/seeds

Current outputs:
- analysis/quant_method_refusal_summary.csv
- analysis/quant_method_pairwise_delta.csv
- figures/quant_method_refusal_comparison.pdf

Acceptance checks:
- matched comparisons across at least two methods with reproducible seeds
- discussion includes method sensitivity and uncertainty bounds

## Workstream E: Mechanistic deepening

Status: partial

Current evidence:
- MLP-vs-attention ablation

Target additions:
- activation/logit-level shift diagnostics for refusal tokens
- layer sensitivity map under quantization

Acceptance checks:
- at least one mechanism figure beyond aggregate refusal-rate shifts

## Workstream F: Safety-utility impact framing

Status: partial

Current evidence:
- safety-utility tradeoff figure added

Target additions:
- benign over-refusal estimate and unsafe pass-through estimate on an evaluation slice

Acceptance checks:
- practical impact metric with confidence intervals

## Submission Gate (for stronger venue)

Move from exploratory claim language only after all checks below are met:

- multi-seed stability reported for core effects
- human-vs-judge validation replaces proxy paragraph
- multi-family model coverage completed
- method-robustness comparison completed
- at least one mechanism-level diagnostic added
