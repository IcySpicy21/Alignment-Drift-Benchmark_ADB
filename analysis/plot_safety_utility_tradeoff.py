#!/usr/bin/env python3
"""Generate safety-vs-capability tradeoff figure.

Inputs:
- analysis/refusal_summary.csv
- analysis/capability_control.csv

Output:
- figures/safety_utility_tradeoff.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "analysis"
FIGURES_DIR = ROOT_DIR / "figures"

sns.set_theme(style="whitegrid", font_scale=1.1)
sns.set_palette("colorblind")


def main() -> None:
    refusal_path = ANALYSIS_DIR / "refusal_summary.csv"
    capability_path = ANALYSIS_DIR / "capability_control.csv"
    points_path = ANALYSIS_DIR / "safety_utility_points.csv"
    output_path = FIGURES_DIR / "safety_utility_tradeoff.pdf"

    required_points = {"model", "precision", "refusal_rate", "mmlu", "gsm8k"}
    if points_path.exists():
        merged = pd.read_csv(points_path)
        if not required_points.issubset(merged.columns):
            print("safety_utility_points.csv missing required columns; skipping tradeoff plot.")
            return
    else:
        if not refusal_path.exists() or not capability_path.exists():
            print("Missing refusal_summary.csv or capability_control.csv; skipping tradeoff plot.")
            return

        refusal = pd.read_csv(refusal_path)
        capability = pd.read_csv(capability_path)

        required_refusal = {"model", "precision", "refusal_rate"}
        required_cap = {"model", "precision", "mmlu", "gsm8k"}
        if not required_refusal.issubset(refusal.columns):
            print("refusal_summary.csv missing required columns; skipping tradeoff plot.")
            return
        if not required_cap.issubset(capability.columns):
            print("capability_control.csv missing required columns; skipping tradeoff plot.")
            return

        merged = refusal.merge(capability, on=["model", "precision"], how="inner")
    if merged.empty:
        print("No overlapping rows between refusal and capability data; skipping tradeoff plot.")
        return

    merged["utility_score"] = merged[["mmlu", "gsm8k"]].mean(axis=1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model, gdf in merged.groupby("model"):
        gdf = gdf.copy()
        ax.plot(
            gdf["utility_score"],
            gdf["refusal_rate"],
            marker="o",
            linewidth=2,
            label=model,
        )
        for _, row in gdf.iterrows():
            ax.annotate(str(row["precision"]), (row["utility_score"], row["refusal_rate"]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Utility Score (mean of MMLU and GSM8K, %)" )
    ax.set_ylabel("Refusal Rate")
    ax.set_title("Safety vs Capability Tradeoff")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
