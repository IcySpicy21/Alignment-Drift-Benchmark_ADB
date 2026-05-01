#!/usr/bin/env python3
"""Print log-derived two-proportion p-values and Mistral INT4 OR (canonical evaluation logs).

Uses only evaluation/logs/*.csv whose filenames omit ``_seed``.

With ``--strict``, exit 1 unless tab:refusal matches the log-derived rounded values
(use after aligning paper/paper.tex with the audit).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "evaluation" / "logs"

LOG_CANONICAL_P_ROUNDED = {
    ("google/gemma-2b-it", "int8"): 0.485,
    ("google/gemma-2b-it", "int4"): 0.366,
    ("mistralai/Mistral-7B-Instruct-v0.2", "int8"): 0.510,
    ("mistralai/Mistral-7B-Instruct-v0.2", "int4"): 0.027,
    ("meta-llama/Meta-Llama-3-8B-Instruct", "int8"): 0.866,
    ("meta-llama/Meta-Llama-3-8B-Instruct", "int4"): 0.559,
}
LOG_CANONICAL_OR_MISTRAL_INT4 = 1.44


def load_canonical_results() -> pd.DataFrame:
    paths = sorted(
        p
        for p in glob.glob(str(LOG_DIR / "*.csv"))
        if "_seed" not in Path(p).name and "bak" not in Path(p).name.lower()
    )
    if not paths:
        raise FileNotFoundError(f"No canonical CSV logs under {LOG_DIR}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def two_sample_p_chi2(df: pd.DataFrame, model: str, alt_precision: str) -> float:
    a = df[(df["model"] == model) & (df["precision"] == "fp16")]
    b = df[(df["model"] == model) & (df["precision"] == alt_precision)]
    r1, n1 = int(a["refusal"].sum()), len(a)
    r2, n2 = int(b["refusal"].sum()), len(b)
    table = [[r1, n1 - r1], [r2, n2 - r2]]
    _, p, _, _ = chi2_contingency(table)
    return float(p)


def mistral_int4_odds_ratio(df: pd.DataFrame) -> float:
    m = "mistralai/Mistral-7B-Instruct-v0.2"
    fp = df[(df["model"] == m) & (df["precision"] == "fp16")]
    q4 = df[(df["model"] == m) & (df["precision"] == "int4")]
    a, b = int(fp["refusal"].sum()), len(fp) - int(fp["refusal"].sum())
    c, d = int(q4["refusal"].sum()), len(q4) - int(q4["refusal"].sum())
    return (c / d) / (a / b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if rounded p/OR differ from log-derived canonicals.",
    )
    args = parser.parse_args()

    df = load_canonical_results()
    print("Canonical statistics from evaluation logs (non-seed CSVs):")
    fail = False
    for (model, prec), rounded in LOG_CANONICAL_P_ROUNDED.items():
        p = two_sample_p_chi2(df, model, prec)
        r3 = round(p, 3)
        ok = r3 == rounded
        tag = "OK " if ok else "MISMATCH " if args.strict else "      "
        print(f"{tag}p {model} vs fp16->{prec}: {p:.6f} -> round3={r3} (log canonical={rounded})")
        if args.strict and not ok:
            fail = True
    or_val = mistral_int4_odds_ratio(df)
    r2 = round(or_val, 2)
    ok_or = r2 == LOG_CANONICAL_OR_MISTRAL_INT4
    tag = "OK " if ok_or else "MISMATCH " if args.strict else "      "
    print(f"{tag}Mistral INT4 OR: {or_val:.6f} -> round2={r2} (log canonical={LOG_CANONICAL_OR_MISTRAL_INT4})")
    if args.strict and not ok_or:
        fail = True

    if fail:
        print("\nStrict check failed: align tab:refusal / abstract OR in paper with logs.", file=sys.stderr)
        return 1
    if not args.strict:
        print(
            "\n(Re-run with --strict after updating paper tab:refusal to match logs.)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
