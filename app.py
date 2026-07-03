#!/usr/bin/env python3
"""
Main application
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.generator import generate_dataset, name_to_seed, normalize_name
from src.utils.helpers import load_config


def export_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def generate(cfg: dict) -> pd.DataFrame:
    chef = cfg["groupe"]["chef"]
    normalized = normalize_name(chef)
    seed = name_to_seed(chef)
    df = generate_dataset(seed)

    print("=== Data Generation ===")
    print(f"Group leader     : {chef}")
    print(f"Normalized name  : {normalized}")
    print(f"Seed             : {seed}")
    print(f"Number of students : {len(df)}")
    print()
    print("Preview of the first 5 rows:")
    print(df.head().to_string(index=False))
    print()
    print("Descriptive statistics:")
    print(df.describe().round(2).to_string())
    print()
    print("Distribution of majors:")
    print(df["major"].value_counts().to_string())

    csv_path = export_csv(df, Path("data/generated/students_data.csv"))
    print(f"\nCSV exported: {csv_path}")

    return df


def run_analyses() -> None:
    cfg = load_config()

    print("\n" + "=" * 65)
    print("LAUNCHING ANALYSES")
    print("=" * 65)

    from src.analysis.univariat import run as run_q1, print_report as pr_q1
    from src.analysis.bivariat import run as run_q2, print_report as pr_q2
    from src.analysis.clustering import run as run_q3, print_report as pr_q3
    from src.analysis.classification import run as run_q4, print_report as pr_q4

    print("\n Univariate analysis ")
    pr_q1(run_q1(cfg))

    print("\n Bivariate analysis ")
    pr_q2(run_q2(cfg))

    print("\n Unsupervised classification ")
    pr_q3(run_q3(cfg))

    print("\nSupervised classification ")
    pr_q4(run_q4(cfg))

    print("\n" + "=" * 65)
    print("ALL ANALYSES COMPLETED")
    print("Results in output/ — Report in docs/rapport.md")
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete Application")
    parser.add_argument("--generate-only", action="store_true", help="Generate data without running analyses")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    generate(cfg)
    if not args.generate_only:
        run_analyses()


if __name__ == "__main__":
    main()