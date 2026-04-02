#!/usr/bin/env python3
"""Prepare a stratified human-annotation batch from evaluation logs.

This script reads seeded evaluation CSVs, maps existing judge/refusal outcomes to
`judge_label`, and writes a human-labeling sheet with blank `human_label`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "model",
    "precision",
    "seed",
    "temperature",
    "category",
    "prompt",
    "response",
    "refusal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stratified judge-annotation batch")
    parser.add_argument(
        "--glob",
        default="evaluation/logs/results_*_seed*_temp*.csv",
        help="Glob pattern for evaluation CSVs",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Target number of rows in annotation batch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--out",
        default="analysis/judge_annotations.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def _read_non_pointer_csvs(paths: Iterable[Path]) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    for path in sorted(paths):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not REQUIRED_COLUMNS.issubset(df.columns):
            continue
        df = df.copy()
        df["_source_file"] = str(path)
        dfs.append(df)
    return dfs


def _stratified_sample(df: pd.DataFrame, n: int, random_seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    group_cols = ["category", "precision"]
    groups = list(df.groupby(group_cols, dropna=False))
    if not groups:
        return df.sample(n=n, random_state=random_seed).reset_index(drop=True)

    per_group = {key: 0 for key, _ in groups}
    capacities = {key: len(gdf) for key, gdf in groups}

    remaining = n
    # First pass: even floor allocation.
    base = n // len(groups)
    for key, _ in groups:
        take = min(base, capacities[key])
        per_group[key] += take
        remaining -= take

    # Second pass: distribute remaining one by one where capacity allows.
    group_order = [key for key, _ in groups]
    i = 0
    while remaining > 0:
        key = group_order[i % len(group_order)]
        if per_group[key] < capacities[key]:
            per_group[key] += 1
            remaining -= 1
        i += 1
        if i > 10_000:
            break

    sampled_parts: list[pd.DataFrame] = []
    for key, gdf in groups:
        take = per_group[key]
        if take <= 0:
            continue
        sampled_parts.append(gdf.sample(n=take, random_state=random_seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return sampled


def main() -> None:
    args = parse_args()

    paths = list(Path(".").glob(args.glob))
    dfs = _read_non_pointer_csvs(paths)
    if not dfs:
        raise RuntimeError("No valid evaluation CSVs found for the provided glob pattern.")

    all_rows = pd.concat(dfs, ignore_index=True)

    # Deduplicate exact repeats from reruns while preserving distinct seed/temp cases.
    all_rows = all_rows.drop_duplicates(
        subset=["model", "precision", "seed", "temperature", "category", "prompt", "response"]
    ).reset_index(drop=True)

    sampled = _stratified_sample(all_rows, n=args.n, random_seed=args.seed)

    sampled = sampled.copy()
    sampled["judge_label"] = sampled["refusal"].map(lambda x: "safe" if bool(x) else "unsafe")
    sampled["human_label"] = ""
    sampled["notes"] = ""

    sampled.insert(0, "prompt_id", [f"audit_{i+1:04d}" for i in range(len(sampled))])

    out_cols = [
        "prompt_id",
        "category",
        "model",
        "precision",
        "seed",
        "temperature",
        "judge_label",
        "human_label",
        "prompt",
        "response",
        "notes",
    ]
    out_df = sampled[out_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out_df)} rows")
    print("Strata counts (category x precision):")
    print(out_df.groupby(["category", "precision"], dropna=False).size().to_string())

    unique_precisions = sorted(out_df["precision"].dropna().astype(str).unique().tolist())
    unique_categories = sorted(out_df["category"].dropna().astype(str).unique().tolist())
    if len(unique_precisions) < 2:
        print(
            "WARNING: Annotation batch includes fewer than 2 precision settings. "
            "Ensure int8/int4 logs are present locally if cross-precision validation is required."
        )
    if len(unique_categories) < 3:
        print(
            "WARNING: Annotation batch includes fewer than 3 prompt categories. "
            "This weakens category-level disagreement analysis."
        )


if __name__ == "__main__":
    main()
