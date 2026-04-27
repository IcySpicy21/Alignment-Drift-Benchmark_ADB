# Pre-arXiv mechanical checklist

Use immediately before `bash scripts/package_arxiv.sh`. Kept terse so it does not duplicate `paper/paper.tex`.

## 1) Manuscript internals

- [ ] Model strings unchanged where intentional:
  - `google/gemma-2b-it`
  - `mistralai/Mistral-7B-Instruct-v0.2`
  - `meta-llama/Meta-Llama-3-8B-Instruct`
- [ ] `\input{judge_validation_paragraph.tex}` still present
- [ ] Author email line current

## 2) Files the tarball expects

- [ ] `paper/paper.tex`
- [ ] `paper/references.bib` (includes `@misc{gupta2026adb_artifacts,...}` for the code drop)
- [ ] `paper/judge_validation_paragraph.tex`
- [ ] `figures/heatmap_drift.pdf`
- [ ] `figures/violin_margin.pdf`
- [ ] `figures/safety_utility_tradeoff.pdf`
- [ ] `figures/quant_method_refusal_comparison.pdf`

## 3) Package

```bash
bash scripts/package_arxiv.sh
```

Expect `paper/arxiv_submission/arxiv_source.tar.gz`.

## 4) Tarball inventory

```bash
tar -tzf paper/arxiv_submission/arxiv_source.tar.gz | sort
```

Confirm the four PDFs + TeX + BIB (+ optional `paper.bbl`) appear.

## 5) Cold compile (host with TeX)

```bash
mkdir -p /tmp/adb_arxiv_submit_check
cp paper/arxiv_submission/arxiv_source.tar.gz /tmp/adb_arxiv_submit_check/
cd /tmp/adb_arxiv_submit_check
tar -xzf arxiv_source.tar.gz
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

- [ ] No missing-file errors
- [ ] Floats resolve

## 6) Repo / legal visibility

- [ ] Public clone matches the commit you think it does
- [ ] arXiv “comments” / ancillary fields mention code availability without pasting long README paragraphs
- [ ] License on GitHub matches what arXiv expects

## 7) Claim hygiene (spot-check, don’t paraphrase here)

- [ ] Exploratory language still visible where required
- [ ] Multiplicity-adjusted Mistral headline ($p=0.16$) appears wherever the raw $0.027$ is highlighted
- [ ] No orphaned `\ref{}` to deleted floats

**Upload when:** tarball cold-compiles and CSV literals match the printed tables you intend to defend.
