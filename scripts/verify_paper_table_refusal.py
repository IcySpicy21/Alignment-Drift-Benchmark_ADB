#!/usr/bin/env python3
"""Cross-check Table tab:refusal p-values and Mistral INT4 OR against canonical logs.

Uses only evaluation/logs/*.csv whose filenames omit ``_seed`` (same scope as the
published refusal_summary pipeline when seed ablations are kept separate).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "evaluation" / "logs"

# Rounded values printed in paper/paper.tex (tab:refusal + abstract OR).
EXPECTED_P_ROUNDED = {
    ("google/gemma-2b-it", "int8"): 0.485,
    ("google/gemma-2b-it", "int4"): 0.366,
    ("mistralai/Mistral-7B-Instruct-v0.2", "int8"): 0.510,
    ("mistralai/Mistral-7B-Instruct-v0.2", "int4"): 0.027,
    ("meta-llama/Meta-Llama-3-8B-Instruct", "int8"): 0.866,
    ("meta-llama/Meta-Llama-3-8B-Instruct", "int4"): 0.559,
}
EXPECTED_OR_MISTRAL_INT4 = 1.44  # two decimals; exact ~1.439


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
    df = load_canonical_results()
    fail = False
    for (model, prec), rounded in EXPECTED_P_ROUNDED.items():
        p = two_sample_p_chi2(df, model, prec)
        if round(p, 3) != rounded:
            print(f"FAIL p {model} vs fp16->{prec}: raw={p:.6f} round3={round(p, 3)} expected={rounded}")
            fail = True
        else:
            print(f"OK  p {model} {prec}: {p:.6f} -> {round(p, 3)}")
    or_val = mistral_int4_odds_ratio(df)
    if round(or_val, 2) != EXPECTED_OR_MISTRAL_INT4:
        print(f"FAIL Mistral INT4 OR: {or_val:.6f} round2={round(or_val, 2)} expected={EXPECTED_OR_MISTRAL_INT4}")
        fail = True
    else:
        print(f"OK  Mistral INT4 OR: {or_val:.6f} -> {round(or_val, 2)}")
    if fail:
        print("\nUpdate paper tab:refusal / abstract OR if logs are authoritative.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
