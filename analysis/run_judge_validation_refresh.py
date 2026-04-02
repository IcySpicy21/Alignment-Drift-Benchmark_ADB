#!/usr/bin/env python3
"""Run judge-validation refresh steps in one command.

Pipeline:
1) compute_judge_agreement.py
2) render_judge_validation_paragraph.py
3) compile paper.tex (optional)
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh judge validation outputs and manuscript text")
    parser.add_argument(
        "--input",
        default="analysis/judge_annotations.csv",
        help="Annotation CSV with judge_label and human_label columns",
    )
    parser.add_argument(
        "--outdir",
        default="analysis",
        help="Directory for agreement output CSVs",
    )
    parser.add_argument(
        "--paragraph-output",
        default="paper/judge_validation_paragraph.tex",
        help="Path for rendered LaTeX paragraph",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip pdflatex compile step",
    )
    parser.add_argument(
        "--allow-missing-human-labels",
        action="store_true",
        help="Allow running even if some human_label values are empty",
    )
    parser.add_argument(
        "--label-source",
        choices=["human", "proxy"],
        default="human",
        help="Annotation source descriptor for manuscript wording.",
    )
    return parser.parse_args()


def _validate_annotations(path: Path, allow_missing: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    df = pd.read_csv(path)
    required = {"judge_label", "human_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in annotation file: {missing}")

    non_empty = df["human_label"].astype(str).str.strip().ne("")
    num_non_empty = int(non_empty.sum())
    num_total = len(df)
    num_missing = num_total - num_non_empty

    if num_non_empty == 0:
        raise ValueError("No human_label values found. Fill annotations before running refresh.")

    if num_missing > 0 and not allow_missing:
        raise ValueError(
            f"Found {num_missing} empty human_label values out of {num_total}. "
            "Complete labels or use --allow-missing-human-labels."
        )

    if num_missing > 0 and allow_missing:
        print(
            f"WARNING: proceeding with {num_missing} empty human_label values out of {num_total}."
        )


def main() -> None:
    args = parse_args()

    input_path = ROOT / args.input
    outdir = ROOT / args.outdir
    paragraph_path = ROOT / args.paragraph_output

    _validate_annotations(input_path, allow_missing=args.allow_missing_human_labels)

    _run(
        [
            "python",
            str(ROOT / "analysis" / "compute_judge_agreement.py"),
            "--input",
            str(input_path),
            "--outdir",
            str(outdir),
        ],
        cwd=ROOT,
    )

    _run(
        [
            "python",
            str(ROOT / "analysis" / "render_judge_validation_paragraph.py"),
            "--summary",
            str(outdir / "judge_agreement_summary.csv"),
            "--category",
            str(outdir / "judge_category_disagreement.csv"),
            "--output",
            str(paragraph_path),
            "--label-source",
            args.label_source,
        ],
        cwd=ROOT,
    )

    if not args.skip_compile:
        _run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "paper.tex",
            ],
            cwd=ROOT / "paper",
        )

    print("Judge validation refresh complete.")
    print(f"Input: {input_path}")
    print(f"Outputs: {outdir / 'judge_agreement_summary.csv'}, {outdir / 'judge_confusion_matrix.csv'}, {outdir / 'judge_category_disagreement.csv'}")
    print(f"Paragraph: {paragraph_path}")
    if args.skip_compile:
        print("Compile: skipped")
    else:
        print(f"Compiled: {ROOT / 'paper' / 'paper.pdf'}")


if __name__ == "__main__":
    main()
