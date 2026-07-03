"""
Deterministic dataset generator
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_CONFIG_PATH = Path("config/config.yaml")


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_covariance_matrix(cfg: dict) -> np.ndarray:
    rho = cfg["generation"]["correlation_note_assiduite"]
    std_note = cfg["generation"]["note"]["std"]
    std_assid = cfg["generation"]["assiduite"]["std"]
    cov = rho * std_note * std_assid
    return np.array([[std_note**2, cov], [cov, std_assid**2]])


def generate_dataset(seed: int) -> pd.DataFrame:
    """Generates the full dataset from an integer seed."""
    cfg = _load_config()
    gen_cfg = cfg["generation"]
    ori_cfg = cfg["orientation"]

    rng = np.random.default_rng(seed)

    mean_vector = np.array([gen_cfg["note"]["mean"], gen_cfg["assiduite"]["mean"]])
    cov_matrix = _build_covariance_matrix(cfg)

    samples = rng.multivariate_normal(mean_vector, cov_matrix, size=gen_cfg["n_eleves"])

    notes = np.clip(samples[:, 0], gen_cfg["note"]["min"], gen_cfg["note"]["max"])
    assiduites = np.clip(samples[:, 1], gen_cfg["assiduite"]["min"], gen_cfg["assiduite"]["max"])

    noise = rng.normal(0.0, ori_cfg["bruit_std"], size=len(notes))
    score = ori_cfg["coef_note"] * notes + ori_cfg["coef_assiduite"] * assiduites + noise
    majors = np.where(score >= ori_cfg["seuil"], "scientific", "literary")

    return pd.DataFrame({
        "student_id": [f"E{i:03d}" for i in range(1, gen_cfg["n_eleves"] + 1)],
        "average_grade": np.round(notes, 2),
        "attendance": np.round(assiduites, 1),
        "major": majors,
    })