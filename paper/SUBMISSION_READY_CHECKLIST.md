# Submission-Ready Checklist (Repo-Tied)

Use this checklist right before arXiv upload.

## 1) Manuscript consistency

- [ ] Model IDs are consistent throughout `paper/paper.tex`:
  - `google/gemma-2b-it`
  - `mistralai/Mistral-7B-Instruct-v0.2`
  - `meta-llama/Meta-Llama-3-8B-Instruct`
- [ ] Judge validation paragraph is included via `\input{judge_validation_paragraph.tex}` in `paper/paper.tex`.
- [ ] Author block includes contact email in `paper/paper.tex`.

## 2) Required paper assets exist

- [ ] `paper/paper.tex`
- [ ] `paper/references.bib`
- [ ] `paper/judge_validation_paragraph.tex`
- [ ] `figures/heatmap_drift.pdf`
- [ ] `figures/violin_margin.pdf`
- [ ] `figures/safety_utility_tradeoff.pdf`
- [ ] `figures/quant_method_refusal_comparison.pdf`

## 3) Build and package

Run from repo root:

```bash
bash scripts/package_arxiv.sh
```

Expected output:

- [ ] `paper/arxiv_submission/arxiv_source.tar.gz`

## 4) Verify archive contents

```bash
tar -tzf paper/arxiv_submission/arxiv_source.tar.gz | sort
```

Expected entries:

- [ ] `paper.tex`
- [ ] `references.bib`
- [ ] `judge_validation_paragraph.tex`
- [ ] `heatmap_drift.pdf`
- [ ] `violin_margin.pdf`
- [ ] `safety_utility_tradeoff.pdf`
- [ ] `quant_method_refusal_comparison.pdf`
- [ ] `paper.bbl` (if present locally)

## 5) Local cold compile (recommended)

In a clean temp dir with TeX installed:

```bash
mkdir -p /tmp/adb_arxiv_submit_check
cp paper/arxiv_submission/arxiv_source.tar.gz /tmp/adb_arxiv_submit_check/
cd /tmp/adb_arxiv_submit_check
tar -xzf arxiv_source.tar.gz
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

- [ ] No missing-file errors
- [ ] Tables and figures render cleanly in generated PDF

## 6) Reproducibility links and metadata

- [ ] GitHub URL in `paper/paper.tex` is public and accessible:
  - `https://github.com/maverick0721/Alignment-Drift-Benchmark_ADB`
- [ ] arXiv metadata fields filled:
  - [ ] Category: `cs.AI` (primary)
  - [ ] Cross-list: `cs.LG` (and `cs.CR` if desired)
  - [ ] Comments field includes pages/figures/code URL
  - [ ] License selected intentionally

## 7) Final sanity checks

- [ ] Claims remain explicitly exploratory in abstract, results, and conclusion
- [ ] Corrected non-significance (`adjusted p = 0.16`) is clearly stated
- [ ] No stale references to removed or renamed figures/tables

## Ready-to-upload criterion

Upload when all boxes above are checked and `paper/arxiv_submission/arxiv_source.tar.gz` compiles cleanly from extraction.
