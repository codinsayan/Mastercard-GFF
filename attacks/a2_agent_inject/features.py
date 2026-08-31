"""Cached semantic, economic, coherence, and provenance features for A2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .assets import load_embedding_cache

FEATURE_NAMES = (
    "a2_intent_cart_cosine",
    "a2_amount_vs_cap_ratio",
    "a2_item_count",
    "a2_price_zscore_in_category",
    "a2_max_item_price_share",
    "a2_cart_mutated_post_load",
    "a2_consent_age_hours",
    "a2_cart_coherence",
    "a2_min_item_cosine_to_cart",
)

ROUND0_FEATURES = FEATURE_NAMES[:3]
DURABLE_FEATURES = FEATURE_NAMES[3:]


class Features:
    attack_id = "A2"

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        if "event_id" not in events or "meta" not in events:
            raise ValueError("events must contain event_id and meta columns")
        cache = load_embedding_cache()
        item_vectors: dict[str, np.ndarray] = cache["item_vectors"]  # type: ignore[assignment]
        intent_vectors: dict[str, np.ndarray] = cache["intent_vectors"]  # type: ignore[assignment]
        dimension = int(cache["dimension"])
        rows = [
            self._build_row(event, item_vectors, intent_vectors, dimension)
            for _, event in events.iterrows()
        ]
        result = pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=float)
        result.index = pd.Index(events["event_id"].astype(str), name="event_id")
        return result

    @staticmethod
    def _build_row(
        event: pd.Series,
        item_vectors: dict[str, np.ndarray],
        intent_vectors: dict[str, np.ndarray],
        dimension: int,
    ) -> dict[str, float]:
        meta = event["meta"] if isinstance(event["meta"], dict) else {}
        category = str(meta.get("a2_intent_category", ""))
        items = [str(item) for item in meta.get("a2_cart_items", [])]
        prices = np.asarray(meta.get("a2_item_prices", []), dtype=float)
        medians = np.asarray(meta.get("a2_item_median_prices", []), dtype=float)
        vectors = np.asarray(
            [item_vectors.get(item, np.zeros(dimension, dtype=np.float32)) for item in items],
            dtype=float,
        )
        intent = intent_vectors.get(category, np.zeros(dimension, dtype=np.float32))

        cart_mean = Features._normalized_mean(vectors, dimension)
        intent_cart_cosine = float(np.dot(intent, cart_mean))
        cap = float(meta.get("a2_spend_cap", np.nan))
        amount_vs_cap = float(event.get("amount", np.nan)) / cap if cap > 0 else 0.0
        item_count = float(len(items))

        valid_prices = len(prices) and len(prices) == len(medians) and np.all(medians > 0)
        if valid_prices:
            log_deviations = np.log(np.maximum(prices, 1e-9) / medians) / np.log(2.0)
            price_zscore = float(np.max(np.abs(log_deviations)))
            total = float(prices.sum())
            max_price_share = float(prices.max() / total) if total > 0 else 0.0
        else:
            price_zscore = 0.0
            max_price_share = 0.0

        checkout = pd.Timestamp(event.get("timestamp"))
        consent = pd.to_datetime(meta.get("a2_consent_issued_at"), errors="coerce")
        consent_age = (
            max(float((checkout - consent).total_seconds() / 3600.0), 0.0)
            if not pd.isna(consent)
            else 0.0
        )
        coherence = Features._pairwise_coherence(vectors)
        min_to_cart = Features._minimum_item_to_remainder(vectors, dimension)
        return {
            "a2_intent_cart_cosine": intent_cart_cosine,
            "a2_amount_vs_cap_ratio": amount_vs_cap,
            "a2_item_count": item_count,
            "a2_price_zscore_in_category": price_zscore,
            "a2_max_item_price_share": max_price_share,
            "a2_cart_mutated_post_load": float(
                bool(meta.get("a2_cart_mutated_post_load", False))
            ),
            "a2_consent_age_hours": consent_age,
            "a2_cart_coherence": coherence,
            "a2_min_item_cosine_to_cart": min_to_cart,
        }

    @staticmethod
    def _normalized_mean(vectors: np.ndarray, dimension: int) -> np.ndarray:
        if vectors.size == 0:
            return np.zeros(dimension, dtype=float)
        mean = vectors.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else np.zeros(dimension, dtype=float)

    @staticmethod
    def _pairwise_coherence(vectors: np.ndarray) -> float:
        if len(vectors) < 2:
            return 1.0 if len(vectors) == 1 else 0.0
        similarities = vectors @ vectors.T
        upper = similarities[np.triu_indices(len(vectors), k=1)]
        return float(np.mean(upper))

    @staticmethod
    def _minimum_item_to_remainder(vectors: np.ndarray, dimension: int) -> float:
        if len(vectors) < 2:
            return 1.0 if len(vectors) == 1 else 0.0
        similarities = []
        for index, vector in enumerate(vectors):
            remainder = np.delete(vectors, index, axis=0)
            remainder_mean = Features._normalized_mean(remainder, dimension)
            similarities.append(float(np.dot(vector, remainder_mean)))
        return float(min(similarities))
