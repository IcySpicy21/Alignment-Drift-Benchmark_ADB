# V2 execution plan (~3 weeks)

Calendar for growing the **eval matrix**; narrative for the PDF stays in `paper/paper.tex`.

## Objectives

- Widen checkpoint list (target six public IDs).
- Multiply RNG seeds (five-tuple default).
- Compare INT4 backends where tooling allows.
- Freeze human-reviewed judge agreement.
- Reuse existing margin pipeline for extra diagnostics.

## Matrix sketch

**Models (stretch goal: 6)**

1. `google/gemma-2b-it`
2. `mistralai/Mistral-7B-Instruct-v0.2`
3. `meta-llama/Meta-Llama-3-8B-Instruct`
4. `microsoft/Phi-3-mini-4k-instruct`
5. `Qwen/Qwen2.5-7B-Instruct`
6. `mistralai/Mixtral-8x7B-Instruct-v0.1`

**Precisions:** `fp16`, `int8`, `int4`  
**INT4 methods:** `bitsandbytes` NF4 baseline + AWQ + GPTQ (when containers/deps succeed)  
**Sampling:** seeds `{11,22,33,44,55}`, temps `{0.0,0.7}` (adjust if advisor prefers)

**Metrics logged:** refusal prevalence (global + per tag), margin stats, drift ratio vs FP16, bootstrap summaries after `run_analysis.py` refresh

## 21-day sketch

**Week 1 — Breadth:** finish CLI filename discipline, run FP16→INT8→INT4 for expanding model list, backfill seeds for the original trio, snapshot CSV tarball.  
**Week 2 — Methods + labels:** AWQ/GPTQ pilots, temperature ablation, stratified human batch, first-pass merges.  
**Week 3 — Depth:** failure taxonomy counts, optional logit/layer probes, regenerate all figures, cold LaTeX build, package script smoke test.

(Daily micro-tasks omitted—use your issue tracker.)

## Command reference

Copy/paste loops: `paper/V2_GPU_COMMAND_SHEET.md`  
Single-run template:

```bash
python evaluation/evaluate_alignment_drift.py \
  --models <model_id> \
  --precision <fp16|int8|int4> \
  --seed <seed> \
  --temperature <0.0|0.7>
```

Then:

```bash
python analysis/run_analysis.py
```

## Definition of done (V2 lab milestone)

1. Six-model coverage **or** documented blockers per missing ID.  
2. Five seeds for the three original models (stretch: all six).  
3. Method table with matched prompts/seeds.  
4. Human-agreement artifacts checked in.  
5. Effect statements in PDF align with refreshed CSV + CI columns.  
6. `arxiv_source.tar.gz` rebuild passes cold compile.
