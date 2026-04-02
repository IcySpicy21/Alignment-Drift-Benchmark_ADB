import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "evaluation" / "logs"
ANALYSIS_DIR = ROOT_DIR / "analysis"


def _bootstrap_ci(values, n_samples=2000, alpha=0.05, rng=None):
    arr = np.asarray(values, dtype=float)
    n = arr.size

    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(arr[0]), float(arr[0])

    if rng is None:
        rng = np.random.default_rng(42)

    samples = rng.choice(arr, size=(n_samples, n), replace=True)
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _safe_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1))


def _load_logs():
    files = sorted(glob.glob(str(LOG_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV logs found in {LOG_DIR}")

    dfs = []
    for file_path in files:
        name = Path(file_path).name
        if "bak" in name:
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()

        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            continue

        df = pd.read_csv(file_path)
        df["source_file"] = name

        if "seed" not in df.columns:
            df["seed"] = 42
        if "temperature" not in df.columns:
            df["temperature"] = 0.7

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No usable CSV logs found (files may be backups or LFS pointers).")

    full_df = pd.concat(dfs, ignore_index=True)

    required = {"model", "precision", "refusal", "seed", "temperature"}
    missing = required - set(full_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in logs: {sorted(missing)}")

    full_df["seed"] = full_df["seed"].astype(int)
    full_df["temperature"] = full_df["temperature"].astype(float)
    return full_df


def _build_run_level(df):
    group_cols = ["model", "precision", "seed", "temperature", "source_file"]

    agg = {
        "refusal": ["mean", "count"],
    }

    has_margin = "refusal_margin" in df.columns
    if has_margin:
        agg["refusal_margin"] = ["mean", "std"]

    grouped = df.groupby(group_cols).agg(agg).reset_index()

    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]

    grouped = grouped.rename(
        columns={
            "refusal_mean": "refusal_rate",
            "refusal_count": "n_prompts",
            "refusal_margin_mean": "mean_refusal_margin",
            "refusal_margin_std": "std_refusal_margin",
        }
    )

    if "mean_refusal_margin" not in grouped.columns:
        grouped["mean_refusal_margin"] = np.nan
        grouped["std_refusal_margin"] = np.nan

    fp16 = grouped[grouped["precision"] == "fp16"][
        ["model", "seed", "temperature", "refusal_rate"]
    ].rename(columns={"refusal_rate": "fp16_refusal_rate"})

    merged = grouped.merge(fp16, on=["model", "seed", "temperature"], how="left")
    merged["drift_ratio"] = np.where(
        merged["precision"] == "fp16",
        np.nan,
        (merged["refusal_rate"] - merged["fp16_refusal_rate"]).abs()
        / (merged["fp16_refusal_rate"] + 1e-8),
    )

    return merged


def _aggregate_metric(df, metric_col, rng, n_bootstrap):
    rows = []
    g = df.groupby(["model", "precision", "temperature"])

    for (model, precision, temp), sub in g:
        vals = sub[metric_col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue

        ci_low, ci_high = _bootstrap_ci(vals, n_samples=n_bootstrap, rng=rng)
        rows.append(
            {
                "model": model,
                "precision": precision,
                "temperature": temp,
                "n_seeds": int(vals.size),
                f"{metric_col}_mean": float(vals.mean()),
                f"{metric_col}_std": _safe_std(vals),
                f"{metric_col}_ci95_low": ci_low,
                f"{metric_col}_ci95_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def run_aggregation(bootstrap_samples=2000, random_seed=42):
    rng = np.random.default_rng(random_seed)

    df = _load_logs()
    run_level = _build_run_level(df)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = ANALYSIS_DIR / "seed_run_summary.csv"
    run_level.to_csv(run_path, index=False)

    refusal_agg = _aggregate_metric(run_level, "refusal_rate", rng, bootstrap_samples)
    refusal_path = ANALYSIS_DIR / "seed_aggregate_refusal.csv"
    refusal_agg.to_csv(refusal_path, index=False)

    margin_agg = _aggregate_metric(run_level, "mean_refusal_margin", rng, bootstrap_samples)
    margin_path = ANALYSIS_DIR / "seed_aggregate_margin.csv"
    margin_agg.to_csv(margin_path, index=False)

    drift_agg = _aggregate_metric(
        run_level[run_level["precision"].isin(["int8", "int4"])],
        "drift_ratio",
        rng,
        bootstrap_samples,
    )
    drift_path = ANALYSIS_DIR / "seed_aggregate_drift.csv"
    drift_agg.to_csv(drift_path, index=False)

    print(f"Saved: {run_path}")
    print(f"Saved: {refusal_path}")
    print(f"Saved: {margin_path}")
    print(f"Saved: {drift_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate seed-level stats with bootstrap confidence intervals.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    run_aggregation(bootstrap_samples=args.bootstrap_samples, random_seed=args.random_seed)


if __name__ == "__main__":
    main()
