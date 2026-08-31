"""Reference-calibrated legitimate payment population generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.schema import coerce, validate


@dataclass(frozen=True)
class PopulationConfig:
    customers: int = 20_000
    merchants: int = 2_000
    vpas: int = 5_000
    days: int = 90
    events: int = 3_000_000
    decline_rate: float = 0.04


class BasePopulation:
    """Generate deterministic honest traffic fitted to reference amounts and MCCs."""

    def __init__(self, reference: pd.DataFrame, config: PopulationConfig | None = None):
        if "amount" not in reference:
            raise ValueError("Reference data must contain an 'amount' column")
        normalized = reference.copy()
        normalized["amount"] = pd.to_numeric(normalized["amount"], errors="coerce")
        normalized = normalized.loc[
            np.isfinite(normalized["amount"]) & normalized["amount"].gt(0)
        ]
        if normalized.empty:
            raise ValueError("Reference data must contain at least one positive amount")
        if "mcc" not in normalized:
            normalized["mcc"] = "UNKNOWN"
        normalized["mcc"] = normalized["mcc"].fillna("UNKNOWN").astype(str)

        self.reference = normalized[["amount", "mcc"]].reset_index(drop=True)
        self.config = config or PopulationConfig()
        logged = np.log(self.reference["amount"].to_numpy(dtype=float))
        self._log_amount_mean = float(logged.mean())
        self._log_amount_std = max(float(logged.std()), 0.25)
        counts = self.reference["mcc"].value_counts(normalize=True)
        self._mcc_values = counts.index.to_numpy(dtype=str)
        self._mcc_probabilities = counts.to_numpy(dtype=float)

    @classmethod
    def from_path(cls, path: str | Path, config: PopulationConfig | None = None):
        path = Path(path)
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        aliases = {
            "amt": "amount",
            "TransactionAmt": "amount",
            "category": "mcc",
            "ProductCD": "mcc",
        }
        frame = frame.rename(
            columns={source: target for source, target in aliases.items() if source in frame}
        )
        if "amount" not in frame:
            raise ValueError("Reference data must contain amount, amt, or TransactionAmt")
        if "mcc" not in frame:
            frame["mcc"] = "UNKNOWN"
        return cls(frame[["amount", "mcc"]], config)

    def generate(self, seed: int = 42, events: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cfg = self.config
        n = cfg.events if events is None else events
        if n <= 0 or min(cfg.customers, cfg.merchants, cfg.vpas, cfg.days) <= 0:
            raise ValueError("population sizes, days, and events must be positive")
        if not 0 <= cfg.decline_rate <= 1:
            raise ValueError("decline_rate must be between 0 and 1")

        activity = rng.lognormal(0.0, 0.7, cfg.customers)
        customers = rng.choice(cfg.customers, n, p=activity / activity.sum())

        merchant_mcc = rng.choice(
            self._mcc_values, cfg.merchants, p=self._mcc_probabilities
        )
        favorite_merchants = rng.integers(0, cfg.merchants, size=(cfg.customers, 6))
        uses_affinity = rng.random(n) < 0.78
        affinity_slot = rng.integers(0, favorite_merchants.shape[1], n)
        merchants = np.where(
            uses_affinity,
            favorite_merchants[customers, affinity_slot],
            rng.integers(0, cfg.merchants, n),
        )

        start = pd.Timestamp("2026-01-01")
        day_weights = np.array(
            [
                1.18 if (start.dayofweek + day) % 7 >= 5 else 1.0
                for day in range(cfg.days)
            ],
            dtype=float,
        )
        days = rng.choice(cfg.days, n, p=day_weights / day_weights.sum())
        hour_weights = np.array(
            [
                0.25, 0.18, 0.14, 0.12, 0.12, 0.18, 0.35, 0.65,
                0.90, 1.00, 1.05, 1.10, 1.15, 1.10, 1.00, 1.05,
                1.20, 1.45, 1.75, 2.00, 1.85, 1.45, 0.90, 0.50,
            ],
            dtype=float,
        )
        hours = rng.choice(24, n, p=hour_weights / hour_weights.sum())
        seconds = hours * 3600 + rng.integers(0, 3600, n)
        timestamps = start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(seconds, unit="s")

        channels = rng.choice(
            ["CNP", "UPI_P2P", "UPI_P2M", "AGENTIC", "CARD_PRESENT"],
            n,
            p=[0.24, 0.20, 0.30, 0.02, 0.24],
        )
        channel_scale = np.select(
            [channels == "UPI_P2P", channels == "AGENTIC", channels == "CARD_PRESENT"],
            [1.25, 1.35, 0.80],
            default=1.0,
        )
        between_customer = min(0.45, self._log_amount_std * 0.45)
        within_customer = np.sqrt(self._log_amount_std**2 - between_customer**2)
        spend_profile = rng.normal(0.0, between_customer, cfg.customers)
        amounts = rng.lognormal(
            self._log_amount_mean + spend_profile[customers], within_customer, n
        ) * channel_scale
        low, high = self.reference["amount"].quantile([0.001, 0.999])
        amounts = np.round(np.clip(amounts, low, high * 2), 2)

        beneficiary = rng.integers(0, cfg.vpas, n)
        is_p2p = channels == "UPI_P2P"
        declined = rng.random(n) < cfg.decline_rate
        customer_age = rng.integers(30, 3651, cfg.customers)
        merchant_age = rng.integers(30, 5001, cfg.merchants)
        vpa_age = rng.integers(7, 3651, cfg.vpas)
        customer_bins = rng.choice(
            ["512345", "522145", "541234", "411111", "437551"], cfg.customers
        )
        alternate_device = rng.random(n) < 0.05
        session_duration = rng.integers(20, 600, n)
        session_duration = np.where(
            channels == "AGENTIC", rng.integers(60, 1801, n), session_duration
        )

        frame = pd.DataFrame(
            {
                "event_id": [f"LEGIT-{seed}-{i:09d}" for i in range(n)],
                "timestamp": timestamps,
                "attack_id": "LEGIT",
                "label": 0,
                "round": 0,
                "customer_id": [f"C{x:05d}" for x in customers],
                "customer_account_age_days": customer_age[customers],
                "device_id": [
                    f"D{x:05d}-{1 + int(alt)}"
                    for x, alt in zip(customers, alternate_device, strict=True)
                ],
                "ip_asn": [f"AS{x}" for x in rng.integers(1000, 99999, n)],
                "session_id": [f"S-{seed}-{i:09d}" for i in range(n)],
                "session_duration_s": session_duration,
                "channel": channels,
                "card_id": [f"CARD{x:05d}" for x in customers],
                "card_bin": customer_bins[customers],
                "vpa": [f"vpa{x % cfg.vpas:05d}@bank" for x in customers],
                "token_id": [
                    f"TOK-{x:05d}" if channel == "AGENTIC" else ""
                    for x, channel in zip(customers, channels, strict=True)
                ],
                "counterparty_id": [
                    f"vpa{x:05d}@bank" if p2p else f"M{merchant:04d}"
                    for x, merchant, p2p in zip(
                        beneficiary, merchants, is_p2p, strict=True
                    )
                ],
                "counterparty_type": np.where(is_p2p, "BENEFICIARY", "MERCHANT"),
                "counterparty_mcc": np.where(is_p2p, "", merchant_mcc[merchants]),
                "counterparty_account_age_days": np.where(
                    is_p2p, vpa_age[beneficiary], merchant_age[merchants]
                ),
                "amount": amounts,
                "is_first_time_payee": (
                    rng.random(n)
                    < np.where(is_p2p, 0.16, np.where(uses_affinity, 0.03, 0.11))
                ).astype("int8"),
                "auth_result": np.where(declined, "DECLINED", "APPROVED"),
                "decline_code": np.where(declined, "DO_NOT_HONOR", ""),
                "meta": [{} for _ in range(n)],
            }
        )
        frame = coerce(frame).sort_values(
            ["timestamp", "event_id"], kind="stable"
        ).reset_index(drop=True)
        validate(frame)
        return frame
