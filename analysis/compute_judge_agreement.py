#!/usr/bin/env python3
"""Compute human-vs-judge agreement metrics from annotation CSVs.

Expected input columns:
- judge_label (required)
- human_label (required)
- category (optional)

Outputs:
- judge_agreement_summary.csv
- judge_confusion_matrix.csv
- judge_category_disagreement.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_label(value: object) -> str:
    return str(value).strip().lower()


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _cohen_kappa(confusion: pd.DataFrame) -> float:
    n = float(confusion.to_numpy().sum())
    if n == 0:
        return 0.0

    po = _safe_div(float(confusion.to_numpy().trace()), n)
    row_marginals = confusion.sum(axis=1).astype(float)
    col_marginals = confusion.sum(axis=0).astype(float)
    pe = _safe_div(float((row_marginals * col_marginals).sum()), n * n)

    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def _macro_f1(confusion: pd.DataFrame) -> float:
    labels = list(confusion.index)
    f1_scores: list[float] = []

    for label in labels:
        tp = float(confusion.loc[label, label])
        fp = float(confusion[label].sum() - tp)
        fn = float(confusion.loc[label].sum() - tp)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    if not f1_scores:
        return 0.0
    return float(sum(f1_scores) / len(f1_scores))


def compute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["judge_label"] = working["judge_label"].map(_normalize_label)
    working["human_label"] = working["human_label"].map(_normalize_label)

    labels = sorted(set(working["judge_label"]).union(set(working["human_label"])))
    confusion = pd.crosstab(
        working["human_label"],
        working["judge_label"],
        rownames=["human_label"],
        colnames=["judge_label"],
        dropna=False,
    )

    confusion = confusion.reindex(index=labels, columns=labels, fill_value=0)

    total = int(confusion.to_numpy().sum())
    accuracy = _safe_div(float(confusion.to_numpy().trace()), float(total))
    kappa = _cohen_kappa(confusion)
    macro_f1 = _macro_f1(confusion)

    summary = pd.DataFrame(
        [
            {
                "n": total,
                "accuracy": accuracy,
                "cohen_kappa": kappa,
                "macro_f1": macro_f1,
                "num_labels": len(labels),
            }
        ]
    )

    if "category" in working.columns:
        cat_disagreement = (
            working.assign(disagree=working["judge_label"] != working["human_label"])
            .groupby("category", dropna=False)
            .agg(
                n=("disagree", "size"),
                disagreement_rate=("disagree", "mean"),
            )
            .reset_index()
            .sort_values(by="disagreement_rate", ascending=False)
        )
    else:
        cat_disagreement = pd.DataFrame(
            [
                {
                    "category": "all",
                    "n": len(working),
                    "disagreement_rate": float((working["judge_label"] != working["human_label"]).mean()),
                }
            ]
        )

    confusion_out = confusion.reset_index()
    return summary, confusion_out, cat_disagreement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute judge agreement metrics from annotations")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to annotation CSV containing judge_label and human_label columns",
    )
    parser.add_argument(
        "--outdir",
        default="analysis",
        help="Directory for output CSV files (default: analysis)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required = {"judge_label", "human_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary, confusion, category_disagreement = compute_metrics(df)

    summary_path = outdir / "judge_agreement_summary.csv"
    confusion_path = outdir / "judge_confusion_matrix.csv"
    category_path = outdir / "judge_category_disagreement.csv"

    summary.to_csv(summary_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    category_disagreement.to_csv(category_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {confusion_path}")
    print(f"Wrote {category_path}")


if __name__ == "__main__":
    main()
