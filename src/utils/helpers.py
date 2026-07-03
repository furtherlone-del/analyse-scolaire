"""
Utility functions — Configuration and data loading.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

_CONFIG_PATH = Path("config/config.yaml")
_DATA_PATH = Path("data/generated/students_data.csv")


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Loads the configuration file from the specified path."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: Path = _DATA_PATH) -> pd.DataFrame:
    """Loads the student dataset from the specified CSV path."""
    return pd.read_csv(path)