"""Module de génération déterministe des données — Annexe obligatoire du TP."""

from .hasher import name_to_seed, normalize_name
from .data_generator import generate_dataset

__all__ = ["name_to_seed", "normalize_name", "generate_dataset"]
