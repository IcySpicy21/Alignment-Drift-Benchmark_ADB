# Alignment-Drift-Benchmark_ADB

Bench harness used in IIT Roorkee work on **refusal rates versus weight precision** (FP16 / INT8 / INT4). The goal is not headline accuracy alone: logs also track how often a fixed judge tags a completion as refusal versus compliance when only the checkpoint bit-width changes.

## Quick Start

```bash
pip install -r requirements.txt
```

Evaluate a single precision setting:

```bash
python evaluation/evaluate_alignment_drift.py --precision fp16
python evaluation/evaluate_alignment_drift.py --precision int8
python evaluation/evaluate_alignment_drift.py --precision int4
```

Aggregate CSVs and rebuild figures:

```bash
python analysis/run_analysis.py
```

## GPTQ path (containerized)

When local `auto-gptq` installs fight your Python/CUDA stack, call the wrapper:

```bash
bash scripts/run_gptq_docker.sh
```

Example override:

```bash
bash scripts/run_gptq_docker.sh \
	"python evaluation/evaluate_alignment_drift.py --models TheBloke/Mistral-7B-Instruct-v0.2-GPTQ --precision int4 --quant-method gptq --seed 21 --temperature 0.7 --max-prompts 25"
```

## Where outputs land

- Raw runs: `evaluation/logs/results_*.csv`
- Aggregates: `analysis/refusal_summary.csv`, `analysis/drift_summary.csv`, `analysis/margin_summary.csv`
- Sanity CSVs: `analysis/refusal_coverage.csv`, `analysis/drift_coverage.csv`, `analysis/margin_coverage.csv`
- Plots (names vary by analysis revision): under `figures/`

## Coverage semantics

- Refusal and margin tables include `fp16`, `int8`, and `int4` whenever the evaluator finished that cell.
- Drift rows exist only for quantized arms; baseline `fp16` is implicit.
- Coverage files mark each `(model, precision)` as `computed`, `skipped`, or `not_applicable`; `fp16` drift is `not_applicable` by construction.

## Manuscript-adjacent paths

- LaTeX root: `paper/paper.tex`
- Frozen headline snapshot: `paper/results_headline_summary.txt`
- arXiv bundle script: `scripts/package_arxiv.sh` → `paper/arxiv_submission/arxiv_source.tar.gz`

## Phase-2 matrix (multi-seed)

```bash
bash scripts/run_v2_matrix.sh
python analysis/run_analysis.py
```

Cap prompts for dry runs:

```bash
MAX_PROMPTS=5 bash scripts/run_v2_matrix.sh
```

Mistral-only quick slice:

```bash
bash scripts/run_mistral_fast_slice.sh
```

## Submission runbook

Operational checklist for packaging, release tagging, and visibility lives in `paper/STAGE8_SUBMISSION_VISIBILITY.md`.
