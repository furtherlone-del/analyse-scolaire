"""
Studies the distribution of the average grade: indicators of central tendency
and dispersion, outlier detection (IQR and Z-score), histogram, and boxplot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.helpers import load_config, load_dataset

OUTPUT_DIR = Path("output/question1")
INDICATOR = "average_grade"
INDICATOR_LABEL = "Average grade (/20)"


def compute_descriptive_stats(series: pd.Series) -> dict:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return {
        "n": int(series.count()),
        "mean": round(series.mean(), 2),
        "median": round(series.median(), 2),
        "std_dev": round(series.std(ddof=1), 2),
        "variance": round(series.var(ddof=1), 2),
        "range": round(series.max() - series.min(), 2),
        "minimum": round(series.min(), 2),
        "maximum": round(series.max(), 2),
        "q1": round(q1, 2),
        "q2": round(series.quantile(0.50), 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
    }


def detect_outliers_iqr(
    df: pd.DataFrame, column: str, multiplier: float = 1.5
) -> tuple[pd.DataFrame, dict]:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    outliers = df.loc[mask].copy()
    outliers["type_iqr"] = np.where(outliers[column] < lower, "low", "high")
    return outliers, {
        "q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2),
        "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
        "multiplier": multiplier,
    }


def detect_outliers_zscore(
    df: pd.DataFrame, column: str, threshold: float = 2.0
) -> tuple[pd.DataFrame, dict]:
    mean = df[column].mean()
    std = df[column].std(ddof=1)
    z_scores = (df[column] - mean) / std
    result = df.copy()
    result["z_score"] = z_scores
    mask = np.abs(z_scores) > threshold
    outliers = result.loc[mask].copy()
    outliers["type_z"] = np.where(outliers["z_score"] < 0, "low", "high")
    return outliers, {"mean": round(mean, 2), "std_dev": round(std, 2), "z_threshold": threshold}


def plot_histogram(series: pd.Series, stats: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(series, bins=15, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(stats["mean"], color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"Mean = {stats['mean']}")
    ax.axvline(stats["median"], color="#55A868", linestyle="-.", linewidth=1.5,
               label=f"Median = {stats['median']}")
    ax.set_title("Distribution of Average Grade — Senior Year Students", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average grade (/20)")
    ax.set_ylabel("Number of students")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_boxplot(series: pd.Series, iqr_bounds: dict, outliers_z: pd.DataFrame, output_path: Path) -> None:
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(series, orientation="vertical", patch_artist=True, widths=0.45,
               showfliers=False, boxprops={"facecolor": "#DDDDDD", "edgecolor": "#333333"},
               medianprops={"color": "#C44E52", "linewidth": 2},
               whiskerprops={"color": "#333333"}, capprops={"color": "#333333"})
    if not outliers_z.empty:
        colors = {"low": "#8172B2", "high": "#CCB974"}
        for _, row in outliers_z.iterrows():
            ax.scatter(1, row[INDICATOR], color=colors.get(row["type_z"], "#333333"),
                       s=80, zorder=5, edgecolors="white", linewidths=0.8)
    ax.axhline(iqr_bounds["lower_bound"], color="#8172B2", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(iqr_bounds["upper_bound"], color="#CCB974", linestyle=":", linewidth=1, alpha=0.6)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8172B2", markersize=8, label="Low outlier (Z < −2)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCB974", markersize=8, label="High outlier (Z > 2)"),
        Line2D([0], [0], color="#8172B2", linestyle=":", label=f"Lower IQR bound = {iqr_bounds['lower_bound']}"),
        Line2D([0], [0], color="#CCB974", linestyle=":", label=f"Upper IQR bound = {iqr_bounds['upper_bound']}"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    ax.set_title("Boxplot — Average grade with outliers (Z-score)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average grade (/20)")
    ax.set_xticks([1])
    ax.set_xticklabels(["Senior Year"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run(cfg: dict) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    notes = df[INDICATOR]

    acfg = cfg["analysis"]
    stats = compute_descriptive_stats(notes)
    outliers_iqr, iqr_bounds = detect_outliers_iqr(df, INDICATOR, acfg["iqr_multiplier"])
    outliers_z, z_params = detect_outliers_zscore(df, INDICATOR, acfg["zscore_threshold"])

    plot_histogram(notes, stats, OUTPUT_DIR / "histogram_average_grade.png")
    plot_boxplot(notes, iqr_bounds, outliers_z, OUTPUT_DIR / "boxplot_average_grade.png")

    summary = {
        "indicator": INDICATOR,
        "indicator_label": INDICATOR_LABEL,
        "descriptive_statistics": stats,
        "iqr_detection": {
            "parameters": iqr_bounds,
            "n_outliers": len(outliers_iqr),
            "students": outliers_iqr[["student_id", INDICATOR, "attendance", "major", "type_iqr"]].to_dict(orient="records"),
        },
        "zscore_detection": {
            "parameters": z_params,
            "n_outliers": len(outliers_z),
            "students": outliers_z[["student_id", INDICATOR, "attendance", "major", "z_score", "type_z"]].round(2).to_dict(orient="records"),
        },
    }

    with open(OUTPUT_DIR / "results_question1.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def print_report(summary: dict) -> None:
    s = summary["descriptive_statistics"]
    iqr = summary["iqr_detection"]
    z = summary["zscore_detection"]

    print("=" * 60)
    print("Analysis of Average Grade ")
    print("=" * 60)
    print(f"\nSample size: {s['n']} students\n")
    print("--- Central Tendency ---")
    print(f"  Mean   : {s['mean']} / 20")
    print(f"  Median : {s['median']} / 20")
    print("\n--- Dispersion ---")
    print(f"  Std Dev : {s['std_dev']}")
    print(f"  Variance: {s['variance']}")
    print(f"  Range   : {s['range']}  (min = {s['minimum']}, max = {s['maximum']})")
    print(f"  Q1      : {s['q1']}")
    print(f"  Q2 (med): {s['q2']}")
    print(f"  Q3      : {s['q3']}")
    print(f"  IQR     : {s['iqr']}")
    print("\n--- Outliers (IQR method) ---")
    p = iqr["parameters"]
    print(f"  Bounds : [{p['lower_bound']} ; {p['upper_bound']}]")
    print(f"  Count  : {iqr['n_outliers']}")
    for e in iqr["students"]:
        print(f"    {e['student_id']} : {e['average_grade']} ({e['type_iqr']})")
    print("\n--- Outliers (Z-score method) ---")
    pz = z["parameters"]
    print(f"  Threshold : |Z| > {pz['z_threshold']}")
    print(f"  Count     : {z['n_outliers']}")
    for e in z["students"]:
        print(f"    {e['student_id']} : {e['average_grade']} (Z = {e['z_score']}, {e['type_z']})")
    print(f"\nGraphs exported to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    cfg = load_config()
    print_report(run(cfg))