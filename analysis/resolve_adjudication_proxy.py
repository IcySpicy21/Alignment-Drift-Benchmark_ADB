#!/usr/bin/env python3
"""Resolve merged annotation disagreements with a transparent proxy rule.

Rule:
- If both annotators agree, keep agreed label.
- If disagreement/empty, use judge_label as fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID = {"safe", "unsafe"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve adjudication using proxy fallback")
    parser.add_argument("--input", default="analysis/judge_annotations_human_merged.csv")
    parser.add_argument("--output", default="analysis/judge_annotations_human_final_proxy.csv")
    return parser.parse_args()


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    for c in ["judge_label", "human_label_a", "human_label_b", "human_label"]:
        if c in df.columns:
            df[c] = _norm(df[c])

    if "human_label" not in df.columns:
        df["human_label"] = ""

    unresolved = df["human_label"].eq("")
    df.loc[unresolved, "human_label"] = df.loc[unresolved, "judge_label"]

    if "notes" not in df.columns:
        df["notes"] = ""
    df["notes"] = df["notes"].fillna("")
    df.loc[unresolved, "notes"] = df.loc[unresolved, "notes"].map(
        lambda x: "proxy_adjudication_fallback" if not str(x).strip() else f"{x};proxy_adjudication_fallback"
    )

    bad = ~df["human_label"].isin(VALID)
    if bad.any():
        raise ValueError("Resolved file contains invalid human_label values")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {out}")
    print(df["human_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
