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
    """Generate honest traffic using empirical amounts/MCCs from a public reference file."""

    def __init__(self, reference: pd.DataFrame, config: PopulationConfig | None = None):
        if "amount" not in reference:
            raise ValueError("Reference data must contain an 'amount' column")
        if reference.empty:
            raise ValueError("Reference data cannot be empty")
        if "mcc" not in reference:
            reference = reference.assign(mcc="UNKNOWN")
        self.reference = reference
        self.config = config or PopulationConfig()

    @classmethod
    def from_path(cls, path: str | Path, config: PopulationConfig | None = None):
        path = Path(path)
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        aliases = {"amt": "amount", "category": "mcc", "merchant": "merchant"}
        frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})
        if "amount" not in frame:
            raise ValueError("Reference data must contain 'amount' or Sparkov's 'amt' column")
        if "mcc" not in frame:
            frame["mcc"] = "UNKNOWN"
        return cls(frame[["amount", "mcc"]].dropna(), config)

    def generate(self, seed: int = 42, events: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cfg, n = self.config, events or self.config.events
        customers = rng.integers(0, cfg.customers, n)
        merchants = rng.integers(0, cfg.merchants, n)
        ref_rows = rng.integers(0, len(self.reference), n)
        timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(
            rng.integers(0, cfg.days * 86_400, n), unit="s"
        )
        declined = rng.random(n) < cfg.decline_rate
        channels = rng.choice(
            ["CNP", "UPI_P2P", "UPI_P2M", "AGENTIC", "CARD_PRESENT"],
            n, p=[0.24, 0.20, 0.30, 0.02, 0.24],
        )
        ref = self.reference.iloc[ref_rows].reset_index(drop=True)
        frame = pd.DataFrame({
            "event_id": [f"LEGIT-{seed}-{i:09d}" for i in range(n)],
            "timestamp": timestamps,
            "attack_id": "LEGIT", "label": 0, "round": 0,
            "customer_id": [f"C{x:05d}" for x in customers],
            "customer_account_age_days": rng.integers(30, 3651, n),
            "device_id": [f"D{x:05d}" for x in customers],
            "ip_asn": [f"AS{x}" for x in rng.integers(1000, 99999, n)],
            "session_id": [f"S-{seed}-{i:09d}" for i in range(n)],
            "session_duration_s": rng.integers(10, 1801, n), "channel": channels,
            "card_id": [f"CARD{x:05d}" for x in customers], "card_bin": "411111",
            "vpa": [f"vpa{x % cfg.vpas:05d}@bank" for x in customers], "token_id": "",
            "counterparty_id": [f"M{x:04d}" for x in merchants],
            "counterparty_type": "MERCHANT", "counterparty_mcc": ref["mcc"].astype(str),
            "counterparty_account_age_days": rng.integers(30, 5001, n),
            "amount": pd.to_numeric(ref["amount"]).clip(lower=0).astype(float),
            "is_first_time_payee": (rng.random(n) < 0.08).astype("int8"),
            "auth_result": np.where(declined, "DECLINED", "APPROVED"),
            "decline_code": np.where(declined, "DO_NOT_HONOR", ""),
            "meta": [{} for _ in range(n)],
        })
        frame = coerce(frame).sort_values("timestamp", kind="stable").reset_index(drop=True)
        validate(frame)
        return frame
