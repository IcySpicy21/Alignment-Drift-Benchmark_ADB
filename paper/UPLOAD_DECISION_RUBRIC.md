# Upload now vs wait ~1 week

Decision aid only; numbers live in the manuscript, not here.

## One-line rule

- **Ship now** if you need a timestamped artifact and accept exploratory limits.
- **Pause** if you are optimizing for a skeptical ML reviewer in one shot.

## Gate A — acceptable v1 drop

All must be true:

- Framing stays non-confirmatory in title/abstract/body.
- Multiplicity-adjusted statistic for the Mistral headline is visible next to the raw value.
- Judge section states automation limits (proxy/human status quo).
- `evaluate_alignment_drift.py` → `run_analysis.py` → LaTeX tables chain reproduces on a fresh clone.

*Internal note:* last audit tagged this gate green.

## Gate B — wait for stronger package

Worth delaying if **≥3** of these are still open:

- ≥5 RNG seeds on core `(model, precision)` cells with summarized variance.
- Human adjudication (not regex proxy) feeding `judge_validation_paragraph.tex`.
- >3 model families in the same matrix.
- ≥2 INT4 backends with matched prompts/seeds in the same table.
- Mechanism figure beyond coarse refusal rates.

*Internal note:* most items were partial last time this file was touched.

## “v2 ready” working definition (lab use)

- Seed aggregates + intervals checked in for the cells you cite.
- Judge paragraph sourced from human merges.
- Method table shows matched backend deltas.
- New diagnostic figure referenced in Methods/Results.
- Abstract mentions one cross-seed-stable sentence **only if** the CSVs back it.

## Strategy

- Deadline-driven: tag v1, branch `v2-strength`.
- Otherwise: close Gate B items, then tag once.
