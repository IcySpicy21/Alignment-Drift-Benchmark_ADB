#!/usr/bin/env python3
"""Render a LaTeX paragraph summarizing judge validation metrics.

Inputs:
- judge_agreement_summary.csv
- judge_category_disagreement.csv

Output:
- paper/judge_validation_paragraph.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def _sanitize_text(value: str) -> str:
    # Minimal escaping for LaTeX special chars used in category names.
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render judge validation paragraph as LaTeX")
    parser.add_argument("--summary", default="analysis/judge_agreement_summary.csv")
    parser.add_argument("--category", default="analysis/judge_category_disagreement.csv")
    parser.add_argument("--output", default="paper/judge_validation_paragraph.tex")
    parser.add_argument(
        "--min-n",
        type=int,
        default=50,
        help="Below this sample size, emit stronger cautionary language.",
    )
    parser.add_argument(
        "--label-source",
        choices=["human", "proxy"],
        default="human",
        help="Annotation source descriptor for manuscript wording.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    category_path = Path(args.category)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        output_path.write_text(
            "\\paragraph{Judge Validation Snapshot.} Agreement summary is pending; run the annotation audit pipeline to populate this section.\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
        return

    summary = pd.read_csv(summary_path)
    if summary.empty:
        output_path.write_text(
            "\\paragraph{Judge Validation Snapshot.} Agreement summary is empty; regenerate validation artifacts before manuscript refresh.\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
        return

    row = summary.iloc[0]
    n = int(row.get("n", 0))
    accuracy = float(row.get("accuracy", 0.0))
    kappa = float(row.get("cohen_kappa", 0.0))
    macro_f1 = float(row.get("macro_f1", 0.0))

    top_slice = "not available"
    top_rate = 0.0
    if category_path.exists():
        category = pd.read_csv(category_path)
        if not category.empty and "disagreement_rate" in category.columns:
            category = category.sort_values(by="disagreement_rate", ascending=False)
            top = category.iloc[0]
            top_slice = _sanitize_text(str(top.get("category", "unknown")))
            top_rate = float(top.get("disagreement_rate", 0.0))

    source_phrase = "human-vs-judge" if args.label_source == "human" else "proxy-vs-judge"

    if args.label_source == "proxy":
        caveat = "Proxy labels are heuristic, so this is diagnostic, not human-calibration evidence."
    else:
        caveat = "These values summarize agreement quality and remain directional under current audit constraints."

    if n < args.min_n:
        caveat = "Sample size is limited, so these agreement values should be treated as directional evidence only."

    text = (
        "\\paragraph{Judge Validation Snapshot.} "
        f"{source_phrase.capitalize()} audit (n={n}) gives accuracy={_pct(accuracy)}, Cohen's $\\kappa$={kappa:.3f}, macro-F1={macro_f1:.3f}, and peak category disagreement of {_pct(top_rate)} ({top_slice}). "
        f"{caveat}\n"
    )

    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
