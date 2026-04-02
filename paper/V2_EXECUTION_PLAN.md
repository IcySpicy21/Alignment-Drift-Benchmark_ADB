# V2 Execution Plan (3 Weeks)

This plan upgrades the current exploratory paper into a stronger workshop/top-tier submission candidate.

## Scope

- Expand model coverage from 3 to 6 models.
- Add multi-seed evaluation (5 seeds per model/precision).
- Add quantization-method comparison (NF4 baseline plus AWQ/GPTQ where feasible).
- Add judge validation with human labels.
- Add mechanism-oriented diagnostics from existing refusal-margin pipeline.

## Target Matrix

### Models (6)

1. google/gemma-2b-it
2. mistralai/Mistral-7B-Instruct-v0.2
3. meta-llama/Meta-Llama-3-8B-Instruct
4. microsoft/Phi-3-mini-4k-instruct
5. Qwen/Qwen2.5-7B-Instruct
6. mistralai/Mixtral-8x7B-Instruct-v0.1

### Precisions

1. fp16
2. int8
3. int4

### Quantization methods

1. NF4 (existing bitsandbytes path in evaluation/evaluate_alignment_drift.py)
2. AWQ (new run path)
3. GPTQ (new run path)

### Sampling controls

1. seed in {11, 22, 33, 44, 55}
2. temperature in {0.0, 0.7}

### Core outcomes

1. Refusal rate (overall + by category)
2. Refusal margin statistics
3. Drift ratio relative to fp16
4. Across-seed mean, std, bootstrap CI

## Daily Milestones (21 Days)

### Week 1: Reproducibility and breadth

Day 1:
- Add CLI args for `--seed` and `--temperature` in evaluation/evaluate_alignment_drift.py.
- Ensure output filenames include seed and temperature suffixes.
- Smoke-test one model at fp16/int8/int4 with one seed.

Day 2:
- Add model list support for 6-model matrix in experiments/run_quantization_experiment.py.
- Add resume/skip safety for per-run artifacts.
- Run 3-model x 3-precision x 1-seed baseline refresh.

Day 3:
- Run all 6 models for fp16 (seed=11, temp=0.7).
- Verify logs in evaluation/logs and integrity checks.

Day 4:
- Run all 6 models for int8 (seed=11, temp=0.7).
- Validate no missing CSV schema fields.

Day 5:
- Run all 6 models for int4 (seed=11, temp=0.7).
- Regenerate summaries with analysis/run_analysis.py.

Day 6:
- Add 4 additional seeds for 3 priority models (Gemma, Mistral, Llama-3-8B).
- Produce first across-seed summary table.

Day 7:
- Add remaining 3 models for all 5 seeds (temp=0.7).
- Freeze Week 1 checkpoint and backup artifacts.

### Week 2: Methods robustness and judge validation

Day 8:
- Add AWQ experiment runner (new script under experiments/).
- Run pilot on Mistral int4 only.

Day 9:
- Add GPTQ experiment runner (new script under experiments/).
- Run pilot on Mistral int4 only.

Day 10:
- Expand AWQ/GPTQ to all 3 original models at int4, seed=11.
- Build method comparison table (NF4 vs AWQ vs GPTQ).

Day 11:
- Run temperature sensitivity at T=0.0 for original 3 models (all precisions, seed=11).
- Compare with T=0.7 tables.

Day 12:
- Sample 100 prompts stratified across categories/precisions for human annotation.
- Prepare annotation protocol and blinded sheet.

Day 13:
- Complete first-pass human labels.
- Compute agreement with judge labels.

Day 14:
- Optional second annotator on 40-sample subset.
- Compute Cohen's kappa and disagreement examples.

### Week 3: Analysis depth and paper polish

Day 15:
- Add failure-mode taxonomy labels: over-refusal false positives, unsafe pass-through, unchanged.
- Summarize rates by category and model.

Day 16:
- Add token-level logit shift analysis for refusal/compliance token groups.
- Compare fp16 vs int4 distributions.

Day 17:
- Add per-layer diagnostic proxy (if full activation tracing is expensive, use selective layer probes for MLP blocks).
- Relate probes to observed refusal-margin shift.

Day 18:
- Regenerate all analysis CSVs/figures and produce final V2 tables.
- Verify consistency between paper claims and analysis files.

Day 19:
- Update manuscript sections: Methods, Judge Validation, Results, Limitations.
- Add stronger error analysis figure/table.

Day 20:
- Full compile cycle: pdflatex + bibtex + pdflatex x2.
- Build arXiv package and cold-compile packaged source.

Day 21:
- Final internal review against reviewer-defense checklist.
- Submit updated arXiv version and prepare workshop draft.

## Experiment Command Template

For copy-paste single-GPU and multi-GPU run blocks, use:

- `paper/V2_GPU_COMMAND_SHEET.md`

Use this pattern for standardized runs:

```bash
python evaluation/evaluate_alignment_drift.py \
  --models <model_id> \
  --precision <fp16|int8|int4> \
  --seed <seed> \
  --temperature <0.0|0.7>
```

Then regenerate summaries:

```bash
python analysis/run_analysis.py
```

## Acceptance Criteria for V2

1. At least 6 models with complete fp16/int8/int4 coverage.
2. At least 5 seeds for all original 3 models (preferably all 6).
3. Method robustness table includes NF4 + AWQ + GPTQ.
4. Judge validation table with agreement metrics is present.
5. Main claim is supported by across-seed effect estimates and corrected significance.
6. Paper text, tables, and analysis CSVs are numerically consistent.
