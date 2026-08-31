"""Trusted Sparkov calibration input for the A2 simulation."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.population import BasePopulation, PopulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPARKOV_REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "sparkov_reference.parquet"
SPARKOV_REFERENCE_ROWS = 1_289_169
SPARKOV_REFERENCE_CATEGORIES = 14
SPARKOV_REFERENCE_SHA256 = "66e87e17c8dae627d81e84e785e7cd7fb79ac5cb54d4fe159875084da35335c4"
SPARKOV_REFERENCE_LABEL = "Sparkov public synthetic transactions"
SPARKOV_REFERENCE_SOURCE = "pointe77/credit-card-transaction"
SPARKOV_REFERENCE_LICENSE = "Apache-2.0"

# Sparkov describes transaction contexts while A2 describes shopping intents. The
# mapping is deliberately conservative and is used only to calibrate the frequency
# of normal intent categories. It is not treated as a ground-truth fraud label.
SPARKOV_TO_A2_CATEGORY = {
    "entertainment": "music",
    "food_dining": "kitchen",
    "gas_transport": "automotive",
    "grocery_net": "kitchen",
    "grocery_pos": "kitchen",
    "health_fitness": "fitness",
    "home": "home_decor",
    "kids_pets": "pet_supplies",
    "misc_net": "office",
    "misc_pos": "gardening",
    "personal_care": "personal_care",
    "shopping_net": "electronics",
    "shopping_pos": "fashion",
    "travel": "travel",
}


@lru_cache(maxsize=1)
def load_sparkov_reference() -> pd.DataFrame:
    """Load and strictly validate the committed normalized Sparkov table."""
    if not SPARKOV_REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing shared Sparkov reference: {SPARKOV_REFERENCE_PATH}. "
            "Restore the committed data file before running A2."
        )
    digest = hashlib.sha256(SPARKOV_REFERENCE_PATH.read_bytes()).hexdigest()
    if digest != SPARKOV_REFERENCE_SHA256:
        raise ValueError("Sparkov reference checksum does not match the trusted commit")

    reference = pd.read_parquet(SPARKOV_REFERENCE_PATH, columns=["amount", "mcc"])
    if len(reference) != SPARKOV_REFERENCE_ROWS:
        raise ValueError(
            f"Expected {SPARKOV_REFERENCE_ROWS:,} Sparkov rows, got {len(reference):,}"
        )
    amounts = pd.to_numeric(reference["amount"], errors="coerce")
    if amounts.isna().any() or not np.isfinite(amounts).all() or not amounts.gt(0).all():
        raise ValueError("Sparkov reference amounts must be finite and positive")
    categories = reference["mcc"].astype("string")
    if categories.isna().any() or categories.nunique() != SPARKOV_REFERENCE_CATEGORIES:
        raise ValueError("Sparkov reference merchant categories are incomplete")
    unknown = set(categories.unique()).difference(SPARKOV_TO_A2_CATEGORY)
    if unknown:
        raise ValueError(f"Unmapped Sparkov merchant categories: {sorted(unknown)}")

    validated = pd.DataFrame({"amount": amounts.astype(float), "mcc": categories.astype(str)})
    return validated


def build_sparkov_population(config: PopulationConfig, *, seed: int = 2026) -> pd.DataFrame:
    """Build honest traffic calibrated to Sparkov amount and category marginals."""
    population = BasePopulation(load_sparkov_reference(), config).generate(seed=seed)
    population.attrs.update(
        {
            "a2_reference_calibrated": True,
            "a2_reference_source": SPARKOV_REFERENCE_LABEL,
            "a2_reference_rows": SPARKOV_REFERENCE_ROWS,
            "a2_reference_sha256": SPARKOV_REFERENCE_SHA256,
            "a2_reference_dataset": SPARKOV_REFERENCE_SOURCE,
            "a2_reference_license": SPARKOV_REFERENCE_LICENSE,
        }
    )
    return population


def a2_category_for_mcc(mcc: object) -> str | None:
    """Return the closest A2 intent category for a Sparkov merchant category."""
    return SPARKOV_TO_A2_CATEGORY.get(str(mcc).strip().lower())
