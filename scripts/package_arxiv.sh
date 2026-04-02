#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/paper/arxiv_submission"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/arxiv_source.tar.gz
rm -f "$OUT_DIR"/paper.tex "$OUT_DIR"/references.bib "$OUT_DIR"/paper.bbl
rm -f "$OUT_DIR"/judge_validation_paragraph.tex
rm -f "$OUT_DIR"/heatmap_drift.pdf "$OUT_DIR"/violin_margin.pdf
rm -f "$OUT_DIR"/safety_utility_tradeoff.pdf "$OUT_DIR"/quant_method_refusal_comparison.pdf

# Copy paper and figures to an arXiv-friendly source directory.
cp "$ROOT_DIR/paper/paper.tex" "$OUT_DIR/paper.tex"
cp "$ROOT_DIR/paper/references.bib" "$OUT_DIR/references.bib"
cp "$ROOT_DIR/paper/judge_validation_paragraph.tex" "$OUT_DIR/judge_validation_paragraph.tex"
cp "$ROOT_DIR/figures/heatmap_drift.pdf" "$OUT_DIR/heatmap_drift.pdf"
cp "$ROOT_DIR/figures/violin_margin.pdf" "$OUT_DIR/violin_margin.pdf"
cp "$ROOT_DIR/figures/safety_utility_tradeoff.pdf" "$OUT_DIR/safety_utility_tradeoff.pdf"
cp "$ROOT_DIR/figures/quant_method_refusal_comparison.pdf" "$OUT_DIR/quant_method_refusal_comparison.pdf"

# Include a prebuilt bibliography if available to avoid BibTeX dependency at upload time.
if [[ -f "$ROOT_DIR/paper/paper.bbl" ]]; then
  cp "$ROOT_DIR/paper/paper.bbl" "$OUT_DIR/paper.bbl"
fi

# Rewrite figure paths for arXiv package layout.
sed -i 's#../figures/heatmap_drift.pdf#heatmap_drift.pdf#g' "$OUT_DIR/paper.tex"
sed -i 's#../figures/violin_margin.pdf#violin_margin.pdf#g' "$OUT_DIR/paper.tex"
sed -i 's#../figures/safety_utility_tradeoff.pdf#safety_utility_tradeoff.pdf#g' "$OUT_DIR/paper.tex"
sed -i 's#../figures/quant_method_refusal_comparison.pdf#quant_method_refusal_comparison.pdf#g' "$OUT_DIR/paper.tex"

(
  cd "$OUT_DIR"
  TAR_INPUTS=(
    paper.tex
    references.bib
    judge_validation_paragraph.tex
    heatmap_drift.pdf
    violin_margin.pdf
    safety_utility_tradeoff.pdf
    quant_method_refusal_comparison.pdf
  )

  if [[ -f paper.bbl ]]; then
    TAR_INPUTS+=(paper.bbl)
  fi

  tar -czf arxiv_source.tar.gz "${TAR_INPUTS[@]}"
)

echo "Created: $OUT_DIR/arxiv_source.tar.gz"
