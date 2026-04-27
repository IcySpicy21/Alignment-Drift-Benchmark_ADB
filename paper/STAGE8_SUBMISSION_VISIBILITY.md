# Stage 8 — Ship checklist (paper, arXiv, repo, comms)

Ops-focused steps; wording kept separate from `paper/paper.tex` to limit duplicate Turnitin hits when uploading a tarball of the whole tree.

## 1. Paper freeze

- [ ] `pdflatex`/`bibtex` cycle clean on `paper/paper.tex`
- [ ] Figures referenced in TeX exist under `figures/`
- [ ] Spot-check `analysis/refusal_summary.csv`, `drift_summary.csv`, `margin_summary.csv` against table literals
- [ ] Coverage CSVs (`*_coverage.csv`) show no surprise `skipped` cells for headline runs
- [ ] Title / abstract / conclusion agree on scope (exploratory framing, FDR caveat)

## 2. arXiv bundle

1. `bash scripts/package_arxiv.sh`
2. Inspect tarball:
   - `paper.tex`, `references.bib`, `judge_validation_paragraph.tex`
   - `heatmap_drift.pdf`, `violin_margin.pdf`, `safety_utility_tradeoff.pdf`, `quant_method_refusal_comparison.pdf`
   - optional `paper.bbl` if you generated it locally
3. Upload `paper/arxiv_submission/arxiv_source.tar.gz`
4. Metadata: primary `cs.CL` (or your advisor’s preference); comments field can point readers to the artifact entry in `references.bib` (URL lives there, not duplicated in body text).
5. Fix compile warnings from arXiv’s TeX runner before clicking submit.
6. Note the arXiv id in your lab notebook / issue tracker (optional: one-line pointer in `README.md` after posting).

## 3. GitHub release

- [ ] `README.md` still documents install + entry commands (already distinct from the PDF narrative)
- [ ] Tag e.g. `v1.0.0` after the paper hash you intend to cite
- [ ] Release notes = changelog bullets (commit hashes, figure refresh date), not a copy-paste abstract

```bash
git add .
git commit -m "Stage 8: freeze manuscript + arxiv bundle"
git tag -a v1.0.0 -m "ADB v1.0.0"
git push origin main --tags
```

## 4. Visibility (optional)

**Day 0:** link the preprint + repo; attach one figure that matches the uploaded PDF (heatmap or violin, not an orphaned legacy plot).

**Days 1–3:** short engineering post: what the harness measures, where logs live, how to rerun `run_analysis.py`—skip re-stating formal claims.

**Week 1:** track stars/forks if useful for your thesis appendix; no obligation to optimize for karma.

## 5. Asset list (current pipeline)

Bundled for arXiv (see `scripts/package_arxiv.sh`):

- `figures/heatmap_drift.pdf`
- `figures/violin_margin.pdf`
- `figures/safety_utility_tradeoff.pdf`
- `figures/quant_method_refusal_comparison.pdf`

Auxiliary diagnostics (may exist locally but are not in the minimal arXiv tar):

- `figures/refusal_plot.pdf`, `figures/drift_plot.pdf`, etc., when `run_analysis.py` emits them

Scratch numbers:

- `paper/results_headline_summary.txt`

## 6. Social blurb (rewrite in your own voice; do not paste the abstract)

Skeleton only:

- One sentence: what the **code** does (eval harness + CSVs).
- One sentence: which **three** HF checkpoints the v1 matrix used.
- Link: repo + preprint.

Example structure (fill blanks yourself):

> Repo `___` runs FP16/INT8/INT4 sweeps and dumps judge-scored CSVs; v1 focused on `___`, `___`, `___`. Preprint: `arXiv:____`.
