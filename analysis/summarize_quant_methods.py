#!/usr/bin/env python3
"""Summarize quantization-method robustness from evaluation logs.

Outputs:
- analysis/quant_method_refusal_summary.csv
- analysis/quant_method_pairwise_delta.csv
- figures/quant_method_refusal_comparison.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT_DIR / "evaluation" / "logs"
ANALYSIS_DIR = ROOT_DIR / "analysis"
FIGURES_DIR = ROOT_DIR / "figures"

sns.set_theme(style="whitegrid", font_scale=1.1)
sns.set_palette("colorblind")


def _normalize_family(model_name: str) -> str:
    name = str(model_name)
    if "/" in name:
        name = name.split("/", 1)[1]
    for suffix in ["-AWQ", "-GPTQ", "-awq", "-gptq", "_AWQ", "_GPTQ"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _infer_method(row: pd.Series) -> str:
    method = str(row.get("quant_method", "")).strip().lower()
    if method:
        return method

    model = str(row.get("model", "")).lower()
    if "awq" in model:
        return "awq"
    if "gptq" in model:
        return "gptq"
    return "bitsandbytes"


def main() -> None:
    files = sorted(LOGS_DIR.glob("results_*.csv"))
    frames = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        required = {"model", "precision", "refusal"}
        if not required.issubset(df.columns):
            continue
        if df.empty:
            continue

        work = df.copy()
        if "seed" not in work.columns:
            work["seed"] = 42
        if "temperature" not in work.columns:
            work["temperature"] = 0.7

        work["quant_method"] = work.apply(_infer_method, axis=1)
        work["model_family"] = work["model"].map(_normalize_family)
        work["source_file"] = path.name
        frames.append(work)

    if not frames:
        print("No compatible evaluation logs found. Skipping quant-method summary.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    int4_df = all_df[all_df["precision"].astype(str).str.lower() == "int4"].copy()

    if int4_df.empty:
        print("No INT4 rows found. Skipping quant-method summary.")
        return

    summary = (
        int4_df.groupby(["model_family", "quant_method", "seed", "temperature"], dropna=False)
        .agg(
            refusal_rate=("refusal", "mean"),
            n_prompts=("refusal", "size"),
        )
        .reset_index()
        .sort_values(["model_family", "seed", "temperature", "quant_method"])
    )

    summary_path = ANALYSIS_DIR / "quant_method_refusal_summary.csv"
    summary.to_csv(summary_path, index=False)

    pairwise = summary.pivot_table(
        index=["model_family", "seed", "temperature"],
        columns="quant_method",
        values="refusal_rate",
        aggfunc="first",
    ).reset_index()

    if "bitsandbytes" in pairwise.columns and "awq" in pairwise.columns:
        pairwise["delta_awq_minus_bitsandbytes"] = pairwise["awq"] - pairwise["bitsandbytes"]
    if "bitsandbytes" in pairwise.columns and "gptq" in pairwise.columns:
        pairwise["delta_gptq_minus_bitsandbytes"] = pairwise["gptq"] - pairwise["bitsandbytes"]

    pairwise_path = ANALYSIS_DIR / "quant_method_pairwise_delta.csv"
    pairwise.to_csv(pairwise_path, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    plot_df = summary.copy()
    plot_df["run"] = (
        plot_df["model_family"].astype(str)
        + " | seed="
        + plot_df["seed"].astype(str)
        + " | T="
        + plot_df["temperature"].astype(str)
    )

    sns.barplot(
        data=plot_df,
        x="run",
        y="refusal_rate",
        hue="quant_method",
        ax=ax,
    )
    ax.set_xlabel("Run")
    ax.set_ylabel("Refusal Rate")
    ax.set_title("INT4 Refusal Rate by Quantization Method")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    figure_path = FIGURES_DIR / "quant_method_refusal_comparison.pdf"
    fig.savefig(figure_path)
    plt.close(fig)

    print(f"Wrote {summary_path}")
    print(f"Wrote {pairwise_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
