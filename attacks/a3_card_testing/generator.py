"""Deterministic generator for A3 agentic card-testing campaigns."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.schema import coerce, validate

CARD_PATH = Path(__file__).with_name("card.json")
INTEGER_PARAMS = {"cards_per_hour", "n_merchants", "cards_per_device", "n_bins"}
BIN_PREFIXES = (
    "411111",
    "437551",
    "512345",
    "522145",
    "541234",
    "555555",
    "601100",
    "353011",
    "356600",
    "400005",
    "424242",
    "444433",
    "510510",
    "520082",
    "530125",
    "543210",
    "550000",
    "601111",
    "622126",
    "676770",
)
MAX_CARDS_PER_HOUR = 60.0


def _load_card() -> dict[str, Any]:
    return json.loads(CARD_PATH.read_text(encoding="utf-8"))


class A3Generator:
    """Inject one simulated hour of agentic card testing into legitimate events."""

    attack_id = "A3"

    def default_params(self) -> dict:
        return dict(_load_card()["default_params"])

    def param_space(self) -> dict[str, tuple[float, float]]:
        return {
            name: (bounds[0], bounds[1])
            for name, bounds in _load_card()["param_space"].items()
        }

    def describe_state(self, params: dict) -> dict:
        card = _load_card()
        return {
            "attack_summary": card["attack_summary"],
            "param_semantics": dict(card["param_semantics"]),
            "defender_signals": dict(card["defender_signals"]),
            "current_params": dict(params),
        }

    def generate(self, params: dict, population: Any, seed: int) -> pd.DataFrame:
        values = self._validated_params(params)
        legitimate, round_value, campaign_start = self._validated_population(population)
        rng = np.random.default_rng(seed)

        merchant_pool = self._merchant_pool(
            legitimate, values["n_merchants"], values["merchant_friction_mix"], rng
        )
        primary_count = values["cards_per_hour"]
        dump_indices = np.arange(primary_count, dtype=int) % values["n_bins"]
        rng.shuffle(dump_indices)
        is_dead = rng.random(primary_count) < values["decline_rate"]
        amounts = np.round(
            rng.uniform(values["amount_min"], values["amount_max"], primary_count), 2
        )
        merchant_indices = rng.integers(0, len(merchant_pool), primary_count)
        spacing_ns = int(3_600_000_000_000 // primary_count)
        bot_campaign_id = f"A3-BOT-{seed}"

        rows: list[dict[str, Any]] = []
        sequence = 0
        for card_index in range(primary_count):
            dump_index = int(dump_indices[card_index])
            bin_prefix = BIN_PREFIXES[dump_index]
            dump_id = f"A3-DUMP-{dump_index:02d}"
            device_index = card_index // values["cards_per_device"]
            device_id = f"A3-DEV-{seed}-{device_index:04d}"
            bot_session_id = f"{bot_campaign_id}-S{device_index:04d}"
            card_id = f"A3-CARD-{seed}-{card_index:06d}"
            customer_id = f"A3-CUSTOMER-{seed}-{card_index:06d}"
            timestamp = campaign_start + pd.to_timedelta(card_index * spacing_ns, unit="ns")
            merchant = merchant_pool[int(merchant_indices[card_index])]
            high_friction = merchant["friction"] == "HIGH"
            approved = not high_friction and not bool(is_dead[card_index])

            rows.append(
                self._event_row(
                    seed=seed,
                    sequence=sequence,
                    timestamp=timestamp,
                    round_value=round_value,
                    customer_id=customer_id,
                    card_id=card_id,
                    bin_prefix=bin_prefix,
                    dump_id=dump_id,
                    device_id=device_id,
                    bot_session_id=bot_session_id,
                    merchant=merchant,
                    amount=float(amounts[card_index]),
                    approved=approved,
                    strong_auth_decline=high_friction,
                    primary_index=card_index,
                    sequence_order=float(card_index),
                )
            )
            sequence += 1

            if high_friction:
                retry_merchant = self._retry_merchant(merchant_pool, merchant, rng)
                retry_amount = self._retry_amount(
                    float(amounts[card_index]),
                    values["amount_min"],
                    values["amount_max"],
                    rng,
                )
                rows.append(
                    self._event_row(
                        seed=seed,
                        sequence=sequence,
                        timestamp=timestamp + pd.Timedelta(seconds=60),
                        round_value=round_value,
                        customer_id=customer_id,
                        card_id=card_id,
                        bin_prefix=bin_prefix,
                        dump_id=dump_id,
                        device_id=device_id,
                        bot_session_id=bot_session_id,
                        merchant=retry_merchant,
                        amount=retry_amount,
                        approved=not bool(is_dead[card_index]),
                        strong_auth_decline=False,
                        primary_index=card_index,
                        sequence_order=card_index + 0.5,
                    )
                )
                sequence += 1

        fraud = pd.DataFrame(rows).sort_values(
            ["timestamp", "_sequence_order"], kind="stable"
        )
        fraud = fraud.reset_index(drop=True)
        fraud["event_id"] = [f"A3-{seed}-{index:09d}" for index in range(len(fraud))]
        fraud["meta"] = [
            {**meta, "a3_attempt_index": index}
            for index, meta in enumerate(fraud["meta"])
        ]
        fraud = fraud.drop(columns=["_sequence_order"])
        fraud = coerce(fraud)

        result = pd.concat([legitimate, fraud], ignore_index=True, sort=False)
        result = result.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
        result = coerce(result)
        validate(result)
        return result

    def utility(self, df: pd.DataFrame) -> float:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        required = {"attack_id", "label", "auth_result", "card_id"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing utility columns: {sorted(missing)}")
        approved = df.loc[
            df["attack_id"].eq(self.attack_id)
            & df["label"].eq(1)
            & df["auth_result"].eq("APPROVED"),
            "card_id",
        ].nunique()
        return float(np.clip(approved / MAX_CARDS_PER_HOUR, 0.0, 1.0))

    def _validated_params(self, params: dict) -> dict[str, int | float]:
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        space = self.param_space()
        if set(params) != set(space):
            missing = sorted(set(space).difference(params))
            unknown = sorted(set(params).difference(space))
            raise ValueError(f"Parameter keys differ; missing={missing}, unknown={unknown}")

        values: dict[str, int | float] = {}
        for name, (low, high) in space.items():
            value = params[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            if not low <= numeric <= high:
                raise ValueError(f"{name} must be within [{low}, {high}]")
            if name in INTEGER_PARAMS:
                if not isinstance(value, Integral) and not numeric.is_integer():
                    raise TypeError(f"{name} must be an integer")
                values[name] = int(numeric)
            else:
                values[name] = numeric
        if values["amount_min"] > values["amount_max"]:
            raise ValueError("amount_min cannot exceed amount_max")
        return values

    @staticmethod
    def _validated_population(population: Any) -> tuple[pd.DataFrame, int, pd.Timestamp]:
        if not isinstance(population, pd.DataFrame):
            raise TypeError("population must be a pandas DataFrame")
        if population.empty:
            raise ValueError("population must contain legitimate history")
        validate(population)
        rounds = population["round"].drop_duplicates()
        if len(rounds) != 1:
            raise ValueError("population must contain exactly one round value")
        if population["timestamp"].isna().any():
            raise ValueError("population timestamps cannot be missing")

        legitimate = population.copy(deep=True)
        legitimate["meta"] = legitimate["meta"].map(dict)
        campaign_start = legitimate["timestamp"].max() + pd.Timedelta(seconds=1)
        return legitimate, int(rounds.iloc[0]), campaign_start

    @staticmethod
    def _merchant_pool(
        population: pd.DataFrame,
        count: int,
        friction_mix: float,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        candidates = population.loc[
            population["counterparty_type"].eq("MERCHANT")
            & population["counterparty_id"].ne("")
        ].drop_duplicates("counterparty_id")
        candidates = candidates.sort_values("counterparty_id", kind="stable")
        selected: list[dict[str, Any]] = []
        if not candidates.empty:
            take = min(count, len(candidates))
            positions = rng.choice(len(candidates), size=take, replace=False)
            for position in positions:
                row = candidates.iloc[int(position)]
                selected.append(
                    {
                        "id": str(row["counterparty_id"]),
                        "mcc": str(row["counterparty_mcc"]),
                        "age": int(row["counterparty_account_age_days"]),
                    }
                )

        existing_ids = set(population["counterparty_id"].astype(str))
        synthetic_index = 0
        while len(selected) < count:
            merchant_id = f"A3-MERCHANT-{synthetic_index:04d}"
            synthetic_index += 1
            if merchant_id in existing_ids:
                continue
            selected.append(
                {
                    "id": merchant_id,
                    "mcc": "5999",
                    "age": int(rng.integers(30, 5001)),
                }
            )

        high_count = math.floor(friction_mix * count + 0.5)
        high_positions = {
            int(position) for position in rng.choice(count, size=high_count, replace=False)
        }
        for index, merchant in enumerate(selected):
            merchant["friction"] = "HIGH" if index in high_positions else "LOW"
        return selected

    @staticmethod
    def _retry_merchant(
        pool: list[dict[str, Any]],
        primary: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        low = [merchant for merchant in pool if merchant["friction"] == "LOW"]
        different = [merchant for merchant in low if merchant["id"] != primary["id"]]
        choices = different or low
        if not choices:
            return primary
        return choices[int(rng.integers(0, len(choices)))]

    @staticmethod
    def _retry_amount(
        primary: float,
        minimum: float,
        maximum: float,
        rng: np.random.Generator,
    ) -> float:
        direction = -1.0 if rng.random() < 0.5 else 1.0
        candidate = round(float(np.clip(primary * (1.0 + 0.05 * direction), minimum, maximum)), 2)
        if candidate == primary:
            candidate = round(
                float(np.clip(primary * (1.0 - 0.05 * direction), minimum, maximum)), 2
            )
        return candidate

    @staticmethod
    def _event_row(
        *,
        seed: int,
        sequence: int,
        timestamp: pd.Timestamp,
        round_value: int,
        customer_id: str,
        card_id: str,
        bin_prefix: str,
        dump_id: str,
        device_id: str,
        bot_session_id: str,
        merchant: dict[str, Any],
        amount: float,
        approved: bool,
        strong_auth_decline: bool,
        primary_index: int,
        sequence_order: float,
    ) -> dict[str, Any]:
        return {
            "event_id": f"A3-TEMP-{seed}-{sequence:09d}",
            "timestamp": timestamp,
            "attack_id": "A3",
            "label": 1,
            "round": round_value,
            "customer_id": customer_id,
            "customer_account_age_days": 365,
            "device_id": device_id,
            "ip_asn": f"AS{64000 + (primary_index % 1000)}",
            "session_id": bot_session_id,
            "session_duration_s": 3600,
            "channel": "CNP",
            "card_id": card_id,
            "card_bin": bin_prefix,
            "vpa": "",
            "token_id": "",
            "counterparty_id": merchant["id"],
            "counterparty_type": "MERCHANT",
            "counterparty_mcc": merchant["mcc"],
            "counterparty_account_age_days": merchant["age"],
            "amount": amount,
            "is_first_time_payee": 1,
            "auth_result": "APPROVED" if approved else "DECLINED",
            "decline_code": (
                "" if approved else "STRONG_AUTH_REQUIRED" if strong_auth_decline else "DO_NOT_HONOR"
            ),
            "meta": {
                "a3_bin_prefix": bin_prefix,
                "a3_dump_id": dump_id,
                "a3_bot_session_id": bot_session_id,
                "a3_merchant_friction_tier": merchant["friction"],
                "a3_attempt_index": sequence,
            },
            "_sequence_order": sequence_order,
        }
