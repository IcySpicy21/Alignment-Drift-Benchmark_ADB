#!/usr/bin/env python3
"""Merge two human annotation sheets and produce adjudication artifacts.

Inputs:
- annotator_a.csv
- annotator_b.csv
- base file with judge labels (default analysis/judge_annotations.csv)

Outputs:
- analysis/judge_annotations_human_merged.csv
- analysis/annotation_packet/adjudication_needed.csv
- analysis/annotation_packet/inter_annotator_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID = {"safe", "unsafe"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two annotation sheets")
    parser.add_argument("--a", default="analysis/annotation_packet/annotator_a.csv")
    parser.add_argument("--b", default="analysis/annotation_packet/annotator_b.csv")
    parser.add_argument("--base", default="analysis/judge_annotations.csv")
    parser.add_argument("--out", default="analysis/judge_annotations_human_merged.csv")
    parser.add_argument("--adjudication", default="analysis/annotation_packet/adjudication_needed.csv")
    parser.add_argument("--summary", default="analysis/annotation_packet/inter_annotator_summary.csv")
    return parser.parse_args()


def _normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def main() -> None:
    args = parse_args()
    a_df = pd.read_csv(args.a)
    b_df = pd.read_csv(args.b)
    base_df = pd.read_csv(args.base)

    for df_name, df in [("annotator_a", a_df), ("annotator_b", b_df)]:
        if "prompt_id" not in df.columns or "human_label" not in df.columns:
            raise ValueError(f"{df_name} must include prompt_id and human_label")

    a = a_df[["prompt_id", "human_label", "notes"]].copy()
    b = b_df[["prompt_id", "human_label", "notes"]].copy() if "notes" in b_df.columns else b_df[["prompt_id", "human_label"]].copy()

    a = a.rename(columns={"human_label": "human_label_a", "notes": "notes_a"})
    b = b.rename(columns={"human_label": "human_label_b", "notes": "notes_b"}) if "notes" in b.columns else b.assign(notes_b="")

    merged = base_df.merge(a, on="prompt_id", how="left").merge(b, on="prompt_id", how="left")

    merged["human_label_a"] = _normalize(merged["human_label_a"])
    merged["human_label_b"] = _normalize(merged["human_label_b"])

    bad_a = (~merged["human_label_a"].isin(VALID)) & (merged["human_label_a"] != "")
    bad_b = (~merged["human_label_b"].isin(VALID)) & (merged["human_label_b"] != "")
    if bad_a.any() or bad_b.any():
        raise ValueError("Invalid labels found. Allowed values: safe, unsafe")

    both_present = (merged["human_label_a"] != "") & (merged["human_label_b"] != "")
    agree = both_present & (merged["human_label_a"] == merged["human_label_b"])
    disagree = both_present & (merged["human_label_a"] != merged["human_label_b"])

    merged["human_label"] = ""
    merged.loc[agree, "human_label"] = merged.loc[agree, "human_label_a"]

    adjud_cols = [
        "prompt_id",
        "category",
        "model",
        "precision",
        "seed",
        "temperature",
        "prompt",
        "response",
        "human_label_a",
        "human_label_b",
        "notes_a",
        "notes_b",
    ]
    adjud = merged.loc[disagree, [c for c in adjud_cols if c in merged.columns]].copy()

    n_pairs = int(both_present.sum())
    n_agree = int(agree.sum())
    percent_agree = (n_agree / n_pairs) if n_pairs else 0.0

    summary = pd.DataFrame([
        {
            "n_rows": len(merged),
            "n_with_two_labels": n_pairs,
            "n_agree": n_agree,
            "n_disagree": int(disagree.sum()),
            "percent_agree": percent_agree,
        }
    ])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.adjudication).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(args.out, index=False)
    adjud.to_csv(args.adjudication, index=False)
    summary.to_csv(args.summary, index=False)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.adjudication}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
