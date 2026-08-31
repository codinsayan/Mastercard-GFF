"""Catalogue and cached-embedding access for the A2 simulation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_PATH = PROJECT_ROOT / "data" / "catalogue.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npz"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INTENT_PHRASES = {
    "camping": "buy camping supplies for an outdoor weekend",
    "kitchen": "buy useful kitchen equipment for home cooking",
    "electronics": "buy personal electronics and computer accessories",
    "fitness": "buy fitness equipment for home workouts",
    "gardening": "buy gardening supplies for caring for plants",
    "office": "buy office equipment for a productive workspace",
    "travel": "buy travel essentials for an upcoming trip",
    "pet_supplies": "buy pet supplies for caring for my animal",
    "baby_care": "buy baby-care essentials for a young child",
    "automotive": "buy automotive accessories for maintaining my car",
    "photography": "buy photography equipment and camera accessories",
    "music": "buy musical instruments and recording equipment",
    "fashion": "buy clothing and fashion accessories",
    "home_decor": "buy home-decor items for furnishing a room",
    "personal_care": "buy personal-care and grooming products",
}


@lru_cache(maxsize=1)
def load_catalogue() -> dict[str, list[dict[str, object]]]:
    """Load and validate the committed product catalogue."""
    raw = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    if set(raw) != set(INTENT_PHRASES):
        raise ValueError("catalogue categories do not match the configured intent categories")
    for category, products in raw.items():
        if len(products) < 2:
            raise ValueError(f"catalogue category {category!r} needs at least two products")
        for product in products:
            if set(product) != {"item", "median_price"}:
                raise ValueError(f"invalid product entry in category {category!r}")
            if not str(product["item"]).strip() or float(product["median_price"]) <= 0:
                raise ValueError(f"invalid product values in category {category!r}")
    return raw


@lru_cache(maxsize=1)
def load_embedding_cache() -> dict[str, object]:
    """Load normalized MiniLM vectors without importing sentence-transformers."""
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBEDDINGS_PATH}. Run `python -m "
            "attacks.a2_agent_inject.build_assets` during precomputation."
        )
    with np.load(EMBEDDINGS_PATH, allow_pickle=False) as cache:
        model_name = str(cache["model_name"].item())
        if model_name != MODEL_NAME:
            raise ValueError(f"unexpected embedding model: {model_name}")
        item_texts = cache["item_texts"].astype(str).tolist()
        item_categories = cache["item_categories"].astype(str).tolist()
        item_prices = cache["item_median_prices"].astype(float)
        item_vectors = cache["item_embeddings"].astype(np.float32)
        intent_categories = cache["intent_categories"].astype(str).tolist()
        intent_texts = cache["intent_texts"].astype(str).tolist()
        intent_vectors = cache["intent_embeddings"].astype(np.float32)

    if item_vectors.ndim != 2 or intent_vectors.ndim != 2:
        raise ValueError("embedding arrays must be two-dimensional")
    if item_vectors.shape[1] != intent_vectors.shape[1]:
        raise ValueError("item and intent embedding dimensions differ")
    if len(item_texts) != len(item_categories) or len(item_texts) != len(item_prices):
        raise ValueError("item cache metadata is misaligned")

    return {
        "model_name": model_name,
        "dimension": item_vectors.shape[1],
        "item_vectors": dict(zip(item_texts, item_vectors, strict=True)),
        "item_categories": dict(zip(item_texts, item_categories, strict=True)),
        "item_median_prices": dict(zip(item_texts, item_prices, strict=True)),
        "intent_vectors": dict(zip(intent_categories, intent_vectors, strict=True)),
        "intent_texts": dict(zip(intent_categories, intent_texts, strict=True)),
    }
