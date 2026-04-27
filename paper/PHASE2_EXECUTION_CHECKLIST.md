# Phase 2 — Upgrade backlog (lab tracker)

Internal task board; keep claims out of this file to avoid echoing Turnitin-visible prose.

## Snapshot

- arXiv packaging: workable under exploratory limits
- Venue-hardening: needs breadth + seeds + human labels
- Blockers: RNG coverage, extra families, real annotators, extra quantizers, deeper diagnostics

## Workstream A — RNG / variance

**Priority:** highest  
**Target:** ≥5 seeds × `(model, precision)` for cells you discuss  
**Commands:** `bash scripts/run_v2_matrix.sh` then `python analysis/run_analysis.py`  
**Done when:** aggregate files exist and directional claims survive the spread

## Workstream B — More checkpoints

**Target:** add e.g. Qwen / Phi / Mixtral-class IDs to the same harness  
**Done when:** summaries include new rows and you compare families, not a singleton

## Workstream C — Human labels

**Target:** ≥200 stratified rows, 2 annotators, merge script  
**Tooling:** see `paper/V2_GPU_COMMAND_SHEET.md` §8  
**Done when:** `judge_validation_paragraph.tex` cites human-merged CSVs

## Workstream D — Quant backend parity

**Target:** matched INT4 runs (NF4 vs AWQ vs GPTQ where installs cooperate)  
**Artifacts:** `analysis/quant_method_*`, `figures/quant_method_refusal_comparison.pdf`  
**Done when:** table states seed/prompt parity explicitly

## Workstream E — Mechanism

**Target:** beyond Table-level deltas (logit slices, layer probes, etc.)  
**Done when:** new figure has a Methods subsection tying it to the evaluator

## Workstream F — Product metrics

**Target:** benign over-refusal vs unsafe pass-through estimates on a fixed slice  
**Done when:** numbers carry intervals, not point quotes

## Venue gate (stricter than arXiv)

Promote language only if:

- seed story closed for cited cells
- human judge paragraph shipped
- multi-family coverage landed
- backend comparison landed
- ≥1 mechanism diagnostic published in PDF
