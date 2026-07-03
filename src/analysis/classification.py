"""
Predicts the major (Scientific / Literary) based on average grade and
attendance. Compares logistic regression and decision tree models.
Includes error analysis, confusion matrix, decision boundary, 
pedagogical risks, and recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from ..utils.helpers import load_config, load_dataset

OUTPUT_DIR = Path("output/question4")
FEATURES = ["average_grade", "attendance"]
TARGET = "major"
RANDOM_STATE = 42


def prepare_data(df: pd.DataFrame, test_size: float = 0.30) -> tuple:
    X = df[FEATURES].values
    y = (df[TARGET] == "scientific").astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_test, y_pred, output_dict=True, target_names=["literary", "scientific"])
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_scientific": round(report["scientific"]["precision"], 4),
        "recall_scientific": round(report["scientific"]["recall"], 4),
        "f1_score_scientific": round(report["scientific"]["f1-score"], 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray, name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Literary", "Scientific"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_decision_boundary(model, scaler, X_train, y_train, X_test, y_test, name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    X_all = np.vstack([X_train, X_test]); y_all = np.hstack([y_train, y_test])
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="RdYlBu_r")
    ax.contour(xx, yy, Z, colors="grey", linewidths=0.5, linestyles="--")
    colors = {0: "#4C72B0", 1: "#C44E52"}
    markers = {0: "o", 1: "s"}
    for label in [0, 1]:
        mask = y_all == label
        ax.scatter(X_all[mask, 1], X_all[mask, 0], marker=markers[label], c=colors[label], alpha=0.7, edgecolors="white", s=45, label=["Literary", "Scientific"][label])
    ax.set_xlabel("Attendance (%)"); ax.set_ylabel("Average grade (/20)")
    ax.set_title(f"Decision Boundary — {name} (normalized space)", fontweight="bold")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def analyze_errors(model, X_test: np.ndarray, y_test: np.ndarray, scaler: StandardScaler) -> dict:
    y_pred = model.predict(X_test)
    X_orig = scaler.inverse_transform(X_test)
    fp_idx = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_test == 1))[0]
    fp_list = [{"grade": round(X_orig[i, 0], 2), "attendance": round(X_orig[i, 1], 1)} for i in fp_idx]
    fn_list = [{"grade": round(X_orig[i, 0], 2), "attendance": round(X_orig[i, 1], 1)} for i in fn_idx]
    n_fp, n_fn = len(fp_list), len(fn_list)
    n_total = len(y_test)
    n_err = n_fp + n_fn
    return {
        "total_test": n_total, "total_errors": n_err, "error_rate_pct": round(n_err / n_total * 100, 1),
        "n_false_positives": n_fp, "false_positives": fp_list,
        "n_false_negatives": n_fn, "false_negatives": fn_list,
        "bias_direction": "The model leans towards the scientific major." if n_fp > n_fn else "The model leans towards the literary major." if n_fn > n_fp else "Errors are balanced.",
        "pedagogical_risk": (
            "Primary risk: a false positive (literary student classified as scientific) "
            "could steer a student toward an unsuitable major. A false negative "
            "(scientific student classified as literary) could deprive them of an "
            "orientation that fits them. The class council retains the final say; "
            "the model must never substitute human judgment."
        ),
    }


def run(cfg: dict) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    acfg = cfg["analysis"]
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, acfg["test_size"])

    log_reg = LogisticRegression(random_state=RANDOM_STATE).fit(X_train, y_train)
    tree = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE).fit(X_train, y_train)

    lr_met = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
    tr_met = evaluate_model(tree, X_test, y_test, "Decision Tree")

    selected = (log_reg, lr_met, "Logistic Regression") if lr_met["accuracy"] >= tr_met["accuracy"] else (tree, tr_met, "Decision Tree")

    plot_confusion_matrix(log_reg, X_test, y_test, "Logistic Regression", OUTPUT_DIR / "confusion_logistic_regression.png")
    plot_confusion_matrix(tree, X_test, y_test, "Decision Tree", OUTPUT_DIR / "confusion_decision_tree.png")
    plot_decision_boundary(selected[0], scaler, X_train, y_train, X_test, y_test, selected[2], OUTPUT_DIR / "decision_boundary.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_tree(tree, feature_names=FEATURES, class_names=["Literary", "Scientific"], filled=True, rounded=True, ax=ax)
    ax.set_title("Decision Tree — Major", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "decision_tree.png", dpi=150); plt.close(fig)

    err = analyze_errors(selected[0], X_test, y_test, scaler)

    summary = {
        "selected_model": selected[2],
        "performances": {
            "logistic_regression": lr_met, "decision_tree": tr_met,
        },
        "error_analysis": err,
        "principal_synthesis": _build_synthesis(selected[1], err),
    }

    with open(OUTPUT_DIR / "results_question4.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _build_synthesis(metrics: dict, err: dict) -> str:
    acc = metrics["accuracy"] * 100
    prec = metrics["precision_scientific"] * 100
    rec = metrics["recall_scientific"] * 100
    return (
        f"This orientation prediction system achieves an accuracy of {acc:.1f}% "
        f"on the test set, with a precision of {prec:.1f}% and a recall of {rec:.1f}% "
        f"for the scientific major. The confusion matrix shows {err['total_errors']} "
        f"errors out of {err['total_test']} students tested ({err['error_rate_pct']}%).\n\n"
        f"{err['pedagogical_risk']}\n\n"
        f"Conclusion: the system is not ready for autonomous use. It can serve "
        f"as a decision-support tool for the pedagogical council, provided that (1) "
        f"the prediction is never used as the sole orientation criterion, (2) cases "
        f"of disagreement with teachers are systematically reviewed, and (3) the "
        f"model is re-evaluated annually. Final decisions rest with the class council."
    )


def print_report(summary: dict) -> None:
    lr = summary["performances"]["logistic_regression"]
    tr = summary["performances"]["decision_tree"]
    err = summary["error_analysis"]
    print("=" * 65)
    print("Supervised Classification of Majors")
    print("=" * 65)
    print(f"\nSelected model: {summary['selected_model']}")
    for name, m in [("Logistic Regression", lr), ("Decision Tree", tr)]:
        cm = m["confusion_matrix"]
        print(f"\n--- {name} ---")
        print(f"  Accuracy  : {m['accuracy']:.2%}")
        print(f"  Precision : {m['precision_scientific']:.2%}, Recall : {m['recall_scientific']:.2%}")
        print(f"  Matrix : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"\n--- Error Analysis ---")
    print(f"  Errors : {err['total_errors']}/{err['total_test']} ({err['error_rate_pct']}%)")
    print(f"  FP={err['n_false_positives']}, FN={err['n_false_negatives']}")
    print(f"\n{summary['principal_synthesis']}")
    print(f"\nGraphs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    cfg = load_config()
    print_report(run(cfg))