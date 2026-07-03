"""
Module for generating a deterministic integer seed from a group leader's name.
"""

from __future__ import annotations

import re
import unicodedata

_BASE = 31
_MODULUS = 2**31 - 1


def normalize_name(full_name: str) -> str:
    """
    Normalize the group leader's full name.

    Steps:
        1. NFD Unicode decomposition (isolates diacritics)
        2. Removal of combining marks (accents)
        3. Conversion to uppercase
        4. Splitting into words
        5. Left half = LASTNAME, right half = firstnames
        6. Reorganization: firstnames + lastname

    Examples:
        "VAMI NEGUEM YVO FREED" → "YVOFREEDVAMINEGUEM"
    """
    decomposed = unicodedata.normalize("NFD", full_name.strip())
    without_accents = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    upper = without_accents.upper()

    words = re.findall(r"[A-Z]+", upper)
    if len(words) < 2:
        return "".join(words)

    n = len(words) // 2
    lastname = "".join(words[:n])
    firstnames = "".join(words[n:])
    return firstnames + lastname


def name_to_seed(full_name: str) -> int:
    """
    Transform the normalized name into a reproducible integer seed.

    Algorithm: Polynomial hashing inspired by Java's String.hashCode().
    Formula: h = (h × 31 + ord(c)) mod (2^31 − 1) for each character c.

    Properties:
        - Deterministic: same input → same seed
        - Sensitive: two distinct names produce different seeds
        - Bounded: result in [0, 2^31 − 2]
    """
    normalized = normalize_name(full_name)
    if not normalized:
        raise ValueError("The normalized name is empty: please check the group leader's name.")

    seed = 0
    for char in normalized:
        seed = (seed * _BASE + ord(char)) % _MODULUS

    return seed