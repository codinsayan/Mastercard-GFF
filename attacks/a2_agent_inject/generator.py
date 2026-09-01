"""Deterministic generator for prompt-injected agentic-commerce events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.schema import coerce, validate

from .assets import INTENT_PHRASES, load_catalogue, load_embedding_cache
from .reference import a2_category_for_mcc

CARD_PATH = Path(__file__).with_name("card.json")
INTEGER_PARAMS = {"n_sessions", "cart_mutated_post_load", "injected_item_count"}


class Generator:
    """Generate matched honest and attacked AI-shopping sessions."""

    attack_id = "A2"

    def __init__(self) -> None:
        self._card = json.loads(CARD_PATH.read_text(encoding="utf-8"))

    def default_params(self) -> dict:
        return dict(self._card["default_params"])

    def param_space(self) -> dict[str, tuple[float, float]]:
        return {
            name: (bounds[0], bounds[1])
            for name, bounds in self._card["param_space"].items()
        }

    def describe_state(self, params: dict) -> dict:
        resolved = self._resolve_params(params)
        return {
            "attack_summary": self._card["attack_summary"],
            "param_semantics": dict(self._card["param_semantics"]),
            "defender_signals": dict(self._card["defender_signals"]),
            "current_params": resolved,
        }

    def generate(self, params: dict, population: Any, seed: int) -> pd.DataFrame:
        """Return schema-valid sessions, deterministic for the supplied seed."""
        resolved = self._resolve_params(params)
        if not isinstance(population, pd.DataFrame) or population.empty:
            raise ValueError("population must be a non-empty pandas DataFrame")

        rng = np.random.default_rng(seed)
        n_sessions = int(resolved["n_sessions"])
        reference_calibrated = bool(population.attrs.get("a2_reference_calibrated", False))
        reference_source = str(population.attrs.get("a2_reference_source", "inline reference"))
        source_indices = rng.choice(len(population), n_sessions, replace=len(population) < n_sessions)
        frame = population.iloc[source_indices].copy().reset_index(drop=True)
        reference_mccs = frame["counterparty_mcc"].astype(str).to_numpy()
        reference_percentiles = (
            pd.to_numeric(frame["amount"], errors="coerce").rank(method="average", pct=True).to_numpy()
        )
        labels = np.zeros(n_sessions, dtype=np.int8)
        labels[: n_sessions // 2] = 1
        rng.shuffle(labels)

        catalogue = load_catalogue()
        cache = load_embedding_cache()
        categories = np.asarray(list(catalogue), dtype=str)
        item_vectors: dict[str, np.ndarray] = cache["item_vectors"]  # type: ignore[assignment]
        intent_vectors: dict[str, np.ndarray] = cache["intent_vectors"]  # type: ignore[assignment]
        item_categories: dict[str, str] = cache["item_categories"]  # type: ignore[assignment]

        amounts: list[float] = []
        metadata: list[dict[str, object]] = []
        counterparties: list[str] = []
        mccs: list[str] = []
        for index, (timestamp, label, reference_mcc, reference_percentile) in enumerate(
            zip(
                frame["timestamp"],
                labels,
                reference_mccs,
                reference_percentiles,
                strict=True,
            )
        ):
            calibrated_category = (
                a2_category_for_mcc(reference_mcc) if reference_calibrated else None
            )
            category = calibrated_category or str(rng.choice(categories))
            normal_products = catalogue[category]
            injection_count = int(resolved["injected_item_count"]) if label else 0
            minimum_final_count = max(3, injection_count + 2)
            final_count = int(rng.integers(minimum_final_count, 6))
            normal_count = final_count - injection_count
            if reference_calibrated:
                chosen = self._select_normal_products(
                    normal_products,
                    count=normal_count,
                    spending_percentile=float(reference_percentile),
                    rng=rng,
                )
            else:
                chosen = rng.choice(len(normal_products), size=normal_count, replace=False)
            cart_items = [str(normal_products[position]["item"]) for position in chosen]
            medians = [float(normal_products[position]["median_price"]) for position in chosen]
            item_prices = [
                round(median * float(rng.lognormal(mean=0.0, sigma=0.08)), 2)
                for median in medians
            ]
            cart_categories = [category] * normal_count
            injected_indices: list[int] = []
            realized_distances: list[float] = []

            if label:
                injected_products = self._select_injected_products(
                    intent_category=category,
                    target_distance=float(resolved["injection_semantic_distance"]),
                    count=injection_count,
                    excluded=set(cart_items),
                    rng=rng,
                    item_vectors=item_vectors,
                    intent_vectors=intent_vectors,
                    item_categories=item_categories,
                )
                for item_name, item_category, median, distance in injected_products:
                    injected_indices.append(len(cart_items))
                    cart_items.append(item_name)
                    cart_categories.append(item_category)
                    medians.append(median)
                    item_prices.append(
                        round(
                            median
                            * float(resolved["price_multiplier"])
                            * float(rng.lognormal(mean=0.0, sigma=0.02)),
                            2,
                        )
                    )
                    realized_distances.append(distance)

            total = round(float(sum(item_prices)), 2)
            if label:
                utilization = float(
                    np.clip(
                        rng.normal(float(resolved["spend_cap_utilization"]), 0.07),
                        0.55,
                        0.96,
                    )
                )
                consent_age = float(resolved["consent_age_hours"])
                mutated = bool(resolved["cart_mutated_post_load"])
            else:
                utilization = float(np.clip(rng.normal(0.80, 0.08), 0.55, 0.96))
                consent_age = float(rng.uniform(0.1, 3.0))
                mutated = False
            spend_cap = round(total / utilization, 2)
            event_time = pd.Timestamp(timestamp)
            consent_time = event_time - pd.Timedelta(hours=consent_age)

            amounts.append(total)
            counterparties.append(f"A2M-{category.upper()}-{index % 25:02d}")
            mccs.append(self._mcc_for_category(category))
            metadata.append(
                {
                    "a2_intent_text": INTENT_PHRASES[category],
                    "a2_intent_category": category,
                    "a2_cart_items": cart_items,
                    "a2_cart_categories": cart_categories,
                    "a2_item_prices": item_prices,
                    "a2_item_median_prices": medians,
                    "a2_injected_index": injected_indices[0] if injected_indices else -1,
                    "a2_injected_indices": injected_indices,
                    "a2_consent_issued_at": consent_time.isoformat(),
                    "a2_cart_mutated_post_load": mutated,
                    "a2_spend_cap": spend_cap,
                    "a2_realized_semantic_distances": realized_distances,
                    "a2_reference_calibrated": reference_calibrated,
                    "a2_reference_source": reference_source,
                    "a2_reference_mcc": reference_mcc if reference_calibrated else "",
                    "a2_reference_amount_percentile": (
                        round(float(reference_percentile), 6) if reference_calibrated else None
                    ),
                }
            )

        frame["event_id"] = [f"A2-{seed}-{index:06d}" for index in range(n_sessions)]
        frame["attack_id"] = np.where(labels == 1, self.attack_id, "LEGIT")
        frame["label"] = labels
        frame["round"] = 0
        frame["session_id"] = [f"A2S-{seed}-{index:06d}" for index in range(n_sessions)]
        frame["session_duration_s"] = rng.integers(90, 1801, n_sessions)
        frame["channel"] = "AGENTIC"
        frame["token_id"] = [f"A2TOK-{seed}-{index:06d}" for index in range(n_sessions)]
        frame["counterparty_id"] = counterparties
        frame["counterparty_type"] = "MERCHANT"
        frame["counterparty_mcc"] = mccs
        frame["amount"] = amounts
        frame["auth_result"] = "APPROVED"
        frame["decline_code"] = ""
        frame["meta"] = metadata
        frame = coerce(frame)
        validate(frame)
        return frame

    @staticmethod
    def _select_normal_products(
        products: list[dict[str, object]],
        *,
        count: int,
        spending_percentile: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select normal products from the Sparkov-calibrated spending tier."""
        prices = np.asarray([float(product["median_price"]) for product in products])
        order = np.argsort(prices, kind="stable")
        ranks = np.empty(len(products), dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(products))
        target = float(np.clip(spending_percentile, 0.0, 1.0))
        weights = np.exp(-np.abs(ranks - target) / 0.22) + 0.02
        return rng.choice(len(products), size=count, replace=False, p=weights / weights.sum())

    def utility(self, df: pd.DataFrame) -> float:
        """Normalize excess injected value against the maximum configured multiplier."""
        if "meta" not in df or "label" not in df:
            raise ValueError("events must contain label and meta columns")
        utilities: list[float] = []
        max_excess_multiple = max(float(self.param_space()["price_multiplier"][1]) - 1.0, 1.0)
        for meta in df.loc[df["label"].eq(1), "meta"]:
            indices = [int(value) for value in meta.get("a2_injected_indices", [])]
            prices = [float(value) for value in meta.get("a2_item_prices", [])]
            medians = [float(value) for value in meta.get("a2_item_median_prices", [])]
            if not indices:
                continue
            excess = sum(max(prices[i] - medians[i], 0.0) for i in indices)
            maximum = sum(medians[i] * max_excess_multiple for i in indices)
            utilities.append(excess / maximum if maximum > 0 else 0.0)
        return float(np.clip(np.mean(utilities) if utilities else 0.0, 0.0, 1.0))

    def _resolve_params(self, params: dict) -> dict:
        if set(params) != set(self.default_params()):
            missing = sorted(set(self.default_params()).difference(params))
            unknown = sorted(set(params).difference(self.default_params()))
            raise ValueError(f"parameter keys differ; missing={missing}, unknown={unknown}")
        resolved: dict[str, int | float] = {}
        for name, (low, high) in self.param_space().items():
            value = float(params[name])
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"parameter {name!r} must be inside [{low}, {high}]")
            resolved[name] = round(value) if name in INTEGER_PARAMS else value
        return resolved

    @staticmethod
    def _select_injected_products(
        *,
        intent_category: str,
        target_distance: float,
        count: int,
        excluded: set[str],
        rng: np.random.Generator,
        item_vectors: dict[str, np.ndarray],
        intent_vectors: dict[str, np.ndarray],
        item_categories: dict[str, str],
    ) -> list[tuple[str, str, float, float]]:
        cache = load_embedding_cache()
        median_prices: dict[str, float] = cache["item_median_prices"]  # type: ignore[assignment]
        intent_vector = intent_vectors[intent_category]
        candidates: list[tuple[float, str, str, float]] = []
        for item_name, vector in item_vectors.items():
            item_category = item_categories[item_name]
            if item_name in excluded:
                continue
            if target_distance <= 0.35 and item_category != intent_category:
                continue
            if target_distance >= 0.55 and item_category == intent_category:
                continue
            distance = float(np.clip(1.0 - np.dot(intent_vector, vector), 0.0, 2.0))
            candidates.append(
                (abs(distance - target_distance), item_name, item_category, distance)
            )
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        if len(candidates) < count:
            raise ValueError("catalogue does not contain enough injection candidates")
        pool_size = min(len(candidates), max(count, 8))
        selected_positions = rng.choice(pool_size, size=count, replace=False)
        selected = [candidates[int(position)] for position in selected_positions]
        return [
            (name, category, float(median_prices[name]), distance)
            for _, name, category, distance in selected
        ]

    @staticmethod
    def _mcc_for_category(category: str) -> str:
        mapping = {
            "camping": "5941",
            "kitchen": "5719",
            "electronics": "5732",
            "fitness": "5941",
            "gardening": "5261",
            "office": "5943",
            "travel": "5947",
            "pet_supplies": "5995",
            "baby_care": "5641",
            "automotive": "5533",
            "photography": "5946",
            "music": "5733",
            "fashion": "5651",
            "home_decor": "5712",
            "personal_care": "5977",
        }
        return mapping[category]
