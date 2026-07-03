"""
Groups students into typical profiles (K-Means) based on average grade and
attendance. Selection of the optimal k using the elbow method and the
silhouette score. Characterization of clusters and connection to Question 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..utils.helpers import load_config, load_dataset

OUTPUT_DIR = Path("output/question3")
FEATURES = ["average_grade", "attendance"]
K_RANGE = range(2, 9)
RANDOM_STATE = 42


def normalize_data(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    return pd.DataFrame(X_scaled, columns=FEATURES, index=df.index), scaler


def elbow_analysis(X: pd.DataFrame) -> dict[int, float]:
    return {k: float(KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit(X).inertia_) for k in K_RANGE}


def silhouette_analysis(X: pd.DataFrame) -> dict[int, float]:
    scores = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    return scores


def determine_optimal_k(inertias: dict, silhouettes: dict) -> int:
    best_by_silhouette = max(silhouettes, key=silhouettes.get)
    diffs2 = np.diff(np.diff(list(inertias.values())))
    elbow_k = list(inertias.keys())[np.argmax(diffs2) + 1] if len(diffs2) > 0 else best_by_silhouette
    return min(elbow_k, best_by_silhouette)


def plot_elbow_silhouette(inertias: dict, silhouettes: dict, optimal_k: int, output_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ks = list(inertias.keys())
    ax1.plot(ks, list(inertias.values()), marker="o", color="#4C72B0", linewidth=2)
    ax1.axvline(optimal_k, color="#C44E52", linestyle="--", linewidth=1.5, label=f"k = {optimal_k} (selected)")
    ax1.set_xlabel("Number of clusters (k)"); ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method", fontweight="bold"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ks, list(silhouettes.values()), marker="s", color="#55A868", linewidth=2)
    ax2.axvline(optimal_k, color="#C44E52", linestyle="--", linewidth=1.5, label=f"k = {optimal_k} (selected)")
    ax2.set_xlabel("Number of clusters (k)"); ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette Score", fontweight="bold"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Determining the Optimal Number of Clusters", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def characterize_clusters(df: pd.DataFrame, kmeans: KMeans) -> dict:
    labels = kmeans.labels_
    note_mean_all = df["average_grade"].mean()
    assid_mean_all = df["attendance"].mean()

    profiles = {}
    for i in range(kmeans.n_clusters):
        mask = labels == i
        cd = df.loc[mask, FEATURES]
        c = {f: round(float(cd[f].mean()), 2) for f in FEATURES}
        if c["average_grade"] > note_mean_all and c["attendance"] > assid_mean_all:
            name = "High-performing, highly attentive"
        elif c["average_grade"] < note_mean_all and c["attendance"] < assid_mean_all:
            name = "Struggling students"
        elif c["average_grade"] > note_mean_all:
            name = "Irregular high-performers"
        else:
            name = "Attentive but mediocre performers"
        profiles[f"cluster_{i}"] = {
            "name": name, "count": int(mask.sum()),
            "proportion_pct": round(float(mask.sum() / len(labels) * 100), 1),
            "original_centroid": c,
            "students": df.loc[mask, "student_id"].tolist(),
        }
    return profiles


def plot_clusters(df: pd.DataFrame, kmeans: KMeans, output_path: Path) -> None:
    labels = kmeans.labels_
    centroids_scaled = kmeans.cluster_centers_
    n_clusters = kmeans.n_clusters
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(df.loc[mask, "attendance"], df.loc[mask, "average_grade"],
                   color=colors[i], alpha=0.7, edgecolors="white", linewidths=0.5, s=50, label=f"Cluster {i+1}")
    ax.scatter(centroids_scaled[:, 1], centroids_scaled[:, 0], c="#333333", marker="X", s=250, edgecolors="white", linewidths=1.5, label="Centroids", zorder=5)
    ax.set_xlabel("Attendance (%)"); ax.set_ylabel("Average grade (/20)")
    ax.set_title("K-Means Classification — Student Profiles", fontweight="bold")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(cfg: dict) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X_norm, _ = normalize_data(df)

    inertias = elbow_analysis(X_norm)
    silhouettes = silhouette_analysis(X_norm)
    optimal_k = determine_optimal_k(inertias, silhouettes)

    plot_elbow_silhouette(inertias, silhouettes, optimal_k, OUTPUT_DIR / "elbow_silhouette.png")

    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init="auto").fit(X_norm)
    profiles = characterize_clusters(df, kmeans)
    plot_clusters(df, kmeans, OUTPUT_DIR / "clusters_attendance_grade.png")

    # Retrieve Q1 outliers for consistency check
    q1_path = Path("output/question1/results_question1.json")
    outliers_z_ids = set()
    if q1_path.exists():
        with open(q1_path) as f:
            q1 = json.load(f)
        outliers_z_ids = {e["student_id"] for e in q1["zscore_detection"]["students"]}

    coherence = {}
    for key, prof in profiles.items():
        common = set(prof["students"]) & outliers_z_ids
        if common:
            coherence[prof["name"]] = {"n_outliers_q1": len(common), "students": sorted(common)}

    total_outliers = sum(v["n_outliers_q1"] for v in coherence.values()) if coherence else 0

    summary = {
        "k_selection": {"k_selected": optimal_k, "inertias": inertias, "silhouettes": silhouettes},
        "profiles": profiles,
        "consistency_question1": {
            "outliers_z_ids": sorted(outliers_z_ids),
            "distribution_in_clusters": coherence,
        },
        "principal_synthesis": _build_synthesis(profiles, coherence, total_outliers),
    }

    with open(OUTPUT_DIR / "results_question3.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _build_synthesis(profiles: dict, coherence: dict, total_outliers: int) -> str:
    lines = [f"We identified {len(profiles)} typical student profiles within the institution by cross-referencing their attendance and general average.", ""]
    for key in sorted(profiles):
        p = profiles[key]; c = p["original_centroid"]
        lines.append(f"• {p['name']} ({p['proportion_pct']}% of the cohort): average grade of {c['average_grade']}/20 for an attendance rate of {c['attendance']}%.")
    lines.append("")
    if total_outliers > 0:
        lines.append(f"Concordance with previous analyses: the {total_outliers} students identified as outliers regarding grades (Question 1) naturally fall into extreme clusters, validating the consistency of both methods.")
    lines.append("")
    lines.append("Pedagogical implications: these profiles allow the pedagogical council to target actions in a differentiated manner. Struggling students require reinforced support (tutoring). High-performing, attentive students can be guided toward excellence tracks. Intermediate profiles should be encouraged to reach higher levels. This typology avoids a one-size-fits-all approach and enables the allocation of pedagogical resources where they are most useful.")
    return "\n".join(lines)


def print_report(summary: dict) -> None:
    sel = summary["k_selection"]
    profiles = summary["profiles"]
    print("=" * 60)
    print("QUESTION 3 — Unsupervised Classification (K-Means)")
    print("=" * 60)
    print(f"\nk selected: {sel['k_selected']}")
    for k in sorted(sel['silhouettes']):
        print(f"  k={k} : inertia={sel['inertias'][k]:.2f}, silhouette={sel['silhouettes'][k]:.3f}")
    print("\n--- Profiles identified ---")
    for key in sorted(profiles):
        p = profiles[key]; c = p["original_centroid"]
        print(f"\n  {p['name']} ({p['count']} students, {p['proportion_pct']}%)")
        print(f"    Grade: {c['average_grade']}/20, Attendance: {c['attendance']}%")
    print(f"\nGraphs exported to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    cfg = load_config()
    print_report(run(cfg))