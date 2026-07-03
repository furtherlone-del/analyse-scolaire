"""
Studies the relationship between attendance and average grade: normality test,
correlation (Pearson / Spearman), OLS linear regression, prediction intervals,
estimation reliability, and counter-examples.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..utils.helpers import load_config, load_dataset

OUTPUT_DIR = Path("output/question2")
X_COL = "attendance"
Y_COL = "average_grade"


def test_normality(series: pd.Series) -> dict:
    stat, p_value = stats.shapiro(series)
    return {"test": "Shapiro-Wilk", "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4), "normal": bool(p_value > 0.05)}


def choose_correlation_method(x: np.ndarray, y: np.ndarray, norm_x: bool, norm_y: bool) -> tuple[str, float, float]:
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    if norm_x and norm_y:
        return "Pearson", round(pearson_r, 4), round(pearson_p, 4)
    return "Spearman", round(spearman_r, 4), round(spearman_p, 4)


def interpret_correlation(r: float, method: str) -> str:
    abs_r = abs(r)
    strength = "weak" if abs_r < 0.30 else "moderate" if abs_r < 0.50 else "moderate to strong" if abs_r < 0.70 else "strong"
    direction = "positive" if r > 0 else "negative"
    return (f"The {method} coefficient is {r:.3f}: {direction} relationship of {strength} strength. "
            f"{'Higher attendance tends to correlate with higher grades.' if r > 0 else 'Inverse relationship observed.'}")


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    n = len(x)
    return {
        "n": n, "slope": round(slope, 4), "intercept": round(intercept, 4),
        "equation": f"grade = {slope:.4f} × attendance + {intercept:.4f}",
        "r_squared": round(r_squared, 4), "r_squared_pct": round(r_squared * 100, 1),
        "pearson_r": round(r_value, 4), "p_value_slope": round(p_value, 6),
        "rmse": round(np.sqrt(ss_res / (n - 2)), 3),
        "residuals": y - y_pred, "y_pred": y_pred,
        "x_mean": round(x.mean(), 2), "ss_x": np.sum((x - x.mean()) ** 2),
        "df": n - 2, "t_critical": round(stats.t.ppf(0.975, n - 2), 3),
    }


def prediction_interval(x_new: np.ndarray, model: dict, x: np.ndarray, confidence: float = 0.95) -> tuple:
    a, b = model["slope"], model["intercept"]
    n, rmse, x_mean, ss_x = model["n"], model["rmse"], model["x_mean"], model["ss_x"]
    t_crit = stats.t.ppf((1 + confidence) / 2, model["df"])
    y_hat = b + a * x_new
    factor = np.sqrt(1 + 1 / n + (x_new - x_mean) ** 2 / ss_x)
    half_width = t_crit * rmse * factor
    return y_hat, y_hat - half_width, y_hat + half_width


def assess_reliability(x: np.ndarray, model: dict, alert_half_width: float = 4.5) -> dict:
    t_crit = stats.t.ppf(0.975, model["df"])
    min_half_width = t_crit * model["rmse"] * np.sqrt(1 + 1 / model["n"])
    x_grid = np.linspace(x.min(), x.max(), 200)
    _, lower, upper = prediction_interval(x_grid, model, x)
    half_widths = (upper - lower) / 2
    reliable_mask = half_widths <= alert_half_width
    reliable_indices = np.where(reliable_mask)[0]
    x_reliable = [round(x_grid[reliable_indices[0]], 1), round(x_grid[reliable_indices[-1]], 1)] if len(reliable_indices) > 0 else None
    x_std = x.std(ddof=1)
    x_mean = model["x_mean"]
    return {
        "pi_half_width_alert": alert_half_width,
        "min_pi_half_width": round(min_half_width, 2),
        "center_pi_half_width": round(half_widths[len(half_widths) // 2], 2),
        "bounds_pi_half_width": {
            "attendance_min": round(x.min(), 1), "half_width": round(half_widths[0], 2),
            "attendance_max": round(x.max(), 1), "half_width_max": round(half_widths[-1], 2),
        },
        "reliable_attendance_zone": x_reliable,
        "globally_unreliable": bool(min_half_width > alert_half_width),
        "extreme_attendance_thresholds": {
            "below": round(max(x.min(), x_mean - 1.5 * x_std), 1),
            "above": round(min(x.max(), x_mean + 1.5 * x_std), 1),
            "interpretation": (
                f"Below {round(max(x.min(), x_mean - 1.5 * x_std), 1)}% attendance, "
                f"the model has few comparable references."
            ),
        },
        "rmse": model["rmse"],
    }


def find_counter_examples(df: pd.DataFrame, model: dict, threshold_sigma: float = 1.5) -> dict:
    residuals = model["residuals"]
    rmse = model["rmse"]
    threshold = threshold_sigma * rmse
    result = df.copy()
    result["residual"] = np.round(residuals, 2)
    result["predicted_grade"] = np.round(model["y_pred"], 2)
    high_att_low_grade = result[(result[X_COL] >= result[X_COL].median()) & (result["residual"] < -threshold)].sort_values("residual")
    low_att_high_grade = result[(result[X_COL] < result[X_COL].median()) & (result["residual"] > threshold)].sort_values("residual", ascending=False)
    cols = ["student_id", X_COL, Y_COL, "predicted_grade", "residual", "major"]
    result["abs_residual"] = np.abs(result["residual"])
    return {
        "residual_threshold": round(threshold, 2),
        "high_attendance_fail": high_att_low_grade[cols].to_dict(orient="records"),
        "low_attendance_succeed": low_att_high_grade[cols].to_dict(orient="records"),
        "top_surprises": result.nlargest(3, "abs_residual")[cols].to_dict(orient="records"),
    }


def plot_scatter_regression(df: pd.DataFrame, model: dict, x: np.ndarray, output_path: Path, threshold_sigma: float = 1.5) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    x_line = np.linspace(x.min() - 2, x.max() + 2, 200)
    y_hat, lower, upper = prediction_interval(x_line, model, x)
    ax.fill_between(x_line, lower, upper, color="#4C72B0", alpha=0.15, label="95% Prediction interval")
    ax.plot(x_line, y_hat, color="#C44E52", linewidth=2, label=f"ŷ = {model['slope']:.3f}x + {model['intercept']:.2f}")
    ax.scatter(df[X_COL], df[Y_COL], color="#4C72B0", alpha=0.7, edgecolors="white", s=45, label="Students")
    residuals = model["residuals"]
    rmse = model["rmse"]
    mask = np.abs(residuals) > threshold_sigma * rmse
    if mask.any():
        ax.scatter(df.loc[mask, X_COL], df.loc[mask, Y_COL], facecolors="none", edgecolors="#CCB974", linewidths=2, s=80, label=f"Outliers (|residual| > {threshold_sigma}×RMSE)")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Average grade (/20)")
    ax.set_title(f"Linear Regression — R² = {model['r_squared_pct']:.1f}%", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    ax2.axhline(0, color="#C44E52", linestyle="--", linewidth=1)
    ax2.axhline(threshold_sigma * rmse, color="#CCB974", linestyle=":", linewidth=1, label=f"+{threshold_sigma}σ = {threshold_sigma * rmse:.2f}")
    ax2.axhline(-threshold_sigma * rmse, color="#CCB974", linestyle=":", linewidth=1, label=f"−{threshold_sigma}σ")
    ax2.scatter(df[X_COL], residuals, color="#4C72B0", alpha=0.7, edgecolors="white", s=45)
    ax2.set_xlabel("Attendance (%)")
    ax2.set_ylabel("Residual (observed − predicted)")
    ax2.set_title("Regression residuals", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.suptitle("Question 2 — Link between attendance and average grade", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(cfg: dict) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    x = df[X_COL].values
    y = df[Y_COL].values
    acfg = cfg["analysis"]

    norm_x = test_normality(df[X_COL])
    norm_y = test_normality(df[Y_COL])
    method, corr_coef, corr_p = choose_correlation_method(x, y, norm_x["normal"], norm_y["normal"])
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    model = fit_linear_regression(x, y)
    reliability = assess_reliability(x, model, acfg["pi_half_width_alert"])
    counter_examples = find_counter_examples(df, model, acfg["residual_threshold_sigma"])
    plot_scatter_regression(df, model, x, OUTPUT_DIR / "scatter_regression_attendance.png")

    summary = {
        "variables": {"x": X_COL, "y": Y_COL},
        "normality": {"attendance": norm_x, "average_grade": norm_y},
        "correlation": {
            "method_used": method, "coefficient": corr_coef, "p_value": corr_p,
            "interpretation": interpret_correlation(corr_coef, method),
            "pearson": {"r": round(pearson_r, 4), "p": round(pearson_p, 4)},
            "spearman": {"rho": round(spearman_r, 4), "p": round(spearman_p, 4)},
        },
        "regression": {
            "equation": model["equation"], "slope": model["slope"],
            "intercept": model["intercept"], "r_squared": model["r_squared"],
            "r_squared_pct": model["r_squared_pct"], "rmse": model["rmse"],
            "p_value_slope": model["p_value_slope"],
        },
        "reliability": reliability,
        "counter_examples": counter_examples,
    }

    with open(OUTPUT_DIR / "results_question2.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def print_report(summary: dict) -> None:
    corr = summary["correlation"]
    reg = summary["regression"]
    fiab = summary["reliability"]
    ce = summary["counter_examples"]

    print("=" * 65)
    print("QUESTION 2 — Bivariate analysis: attendance → average grade")
    print("=" * 65)
    print("\n--- Normality tests (Shapiro-Wilk) ---")
    for var, res in summary["normality"].items():
        print(f"  {var:15s} : W = {res['statistic']}, p = {res['p_value']} → {'normal' if res['normal'] else 'non-normal'}")
    print(f"\n--- Correlation (method used: {corr['method_used']}) ---")
    print(f"  Coefficient : {corr['coefficient']}  (p = {corr['p_value']})")
    print(f"  Pearson  r  : {corr['pearson']['r']}  (p = {corr['pearson']['p']})")
    print(f"  Spearman ρ  : {corr['spearman']['rho']}  (p = {corr['spearman']['p']})")
    print(f"  → {corr['interpretation']}")
    print("\n--- Simple linear regression ---")
    print(f"  Equation : {reg['equation']}")
    print(f"  R²       : {reg['r_squared']} ({reg['r_squared_pct']}%)")
    print(f"  RMSE     : {reg['rmse']} points")
    print(f"  Significant slope : p = {reg['p_value_slope']}")
    print("\n--- Estimation reliability ---")
    print(f"  Min PI half-width (95%) : ±{fiab['min_pi_half_width']} pts")
    print(f"  Individual estimation reliable?  : {'No' if fiab['globally_unreliable'] else 'Yes (central zone)'}")
    bounds = fiab["bounds_pi_half_width"]
    print(f"  PI at bounds : attend. {bounds['attendance_min']}% → ±{bounds['half_width']} pts ; "
          f"attend. {bounds['attendance_max']}% → ±{bounds['half_width_max']} pts")
    print(f"\n--- Counter-examples (residual threshold: ±{ce['residual_threshold']} pts) ---")
    for label, data in [("High attendance, grade below prediction", ce["high_attendance_fail"]),
                        ("Low attendance, grade above prediction", ce["low_attendance_succeed"])]:
        print(f"  {label} :")
        for e in data or [{"student_id": "(none)"}]:
            print(f"    {e['student_id']}")
    print(f"\nGraph exported to: {OUTPUT_DIR}/scatter_regression_attendance.png")


if __name__ == "__main__":
    cfg = load_config()
    print_report(run(cfg))