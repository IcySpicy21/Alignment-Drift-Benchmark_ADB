from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "paper" / "arxiv_submission"

# Flat tarball layout (matches prior arxiv_source.tar.gz).
ARXIV_FILES = [
    "00README.yaml",
    "paper.tex",
    "references.bib",
    "judge_validation_paragraph.tex",
    "paper.bbl",
    "heatmap_drift.pdf",
    "violin_margin.pdf",
    "safety_utility_tradeoff.pdf",
    "quant_method_refusal_comparison.pdf",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create arXiv source bundle for ADB paper")
    parser.add_argument(
        "--output",
        default="dist/arxiv_source.tar.gz",
        help="Output tar.gz path relative to repository root",
    )
    args = parser.parse_args()
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [name for name in ARXIV_FILES if not (SOURCE_DIR / name).is_file()]
    if missing:
        print(
            "ERROR: missing required files under paper/arxiv_submission/:\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
        return 1

    with tarfile.open(out_path, "w:gz") as tar:
        for name in ARXIV_FILES:
            tar.add(SOURCE_DIR / name, arcname=name)

    print(f"Created arXiv source bundle: {out_path}")
    print(f"Included files: {len(ARXIV_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
