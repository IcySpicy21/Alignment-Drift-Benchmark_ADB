# Upload Decision Rubric (Now vs Delay 7-10 Days)

Use this rubric to decide whether to upload immediately or delay for a stronger version.

## Recommendation in one line

- Upload now if your goal is timestamp + feedback on exploratory evidence.
- Delay 7-10 days if your goal is stronger lab/reviewer impact.

## Gate A: Upload now (exploratory preprint)

Upload now if all are true:

- claims remain explicitly exploratory in title/abstract/body
- corrected non-significance is clearly disclosed
- proxy-vs-judge caveat is explicit wherever judge metrics appear
- reproducibility pipeline runs from evaluator to analysis to paper build

Current status: met.

## Gate B: Delay for stronger impact

Delay if you want high-signal reviewer impression and at least 3 of these are currently unmet:

- 5-seed coverage for core model x precision cells
- human-vs-judge (not proxy) with adjudicated annotations
- multi-family expansion beyond current 3-model set
- quantization method robustness with matched seeds (>=2 methods)
- one mechanism-level diagnostic beyond aggregate rates

Current status: mostly unmet (partial progress only).

## 7-10 day target definition

A stronger v2 is reached when all of the following hold:

- seed aggregates populated for core cells with CI
- human-vs-judge paragraph uses label-source=human
- method robustness table includes matched comparisons with deltas
- mechanism section includes one new diagnostic figure
- final abstract includes one stable cross-seed finding statement

## Practical strategy

- If submission urgency exists: upload now as v1 and continue v2 in parallel.
- If no urgency: execute v2 sprint first, then upload once Gate B is largely satisfied.
