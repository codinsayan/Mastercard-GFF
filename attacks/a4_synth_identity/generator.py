"""Deterministic synthetic-identity build-and-bust event generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.schema import coerce, validate

CARD = json.loads(Path(__file__).with_name("card.json").read_text(encoding="utf-8"))


class A4Generator:
    attack_id = "A4"

    def default_params(self) -> dict:
        return dict(CARD["default_params"])

    def param_space(self) -> dict[str, tuple[float, float]]:
        return {name: tuple(bounds) for name, bounds in CARD["param_space"].items()}

    def describe_state(self, params: dict) -> dict:
        return {
            "attack_summary": CARD["attack_summary"],
            "param_semantics": dict(CARD["param_semantics"]),
            "defender_signals": dict(CARD["defender_signals"]),
            "current_params": dict(params),
        }

    def generate(self, params: dict, population: pd.DataFrame, seed: int) -> pd.DataFrame:
        params = self._validate_params(params)
        if population.empty:
            raise ValueError("population cannot be empty")
        required = {
            "counterparty_id",
            "counterparty_mcc",
            "counterparty_account_age_days",
            "card_bin",
            "amount",
        }
        if missing := required.difference(population.columns):
            raise ValueError(f"population is missing columns: {sorted(missing)}")

        rng = np.random.default_rng(seed)
        templates = population.reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        event_number = 0
        build_days = int(params["build_days"])
        overlap = float(params["attribute_overlap_rate"])
        variance = float(params["playbook_variance"])

        for cluster in range(int(params["n_clusters"])):
            cluster_id = f"cluster_{cluster:03d}"
            playbook_id = f"pb_{cluster % max(1, int(params['n_clusters']) // 3):02d}"
            shared_address = f"sunrise-{rng.integers(100, 999)}"
            shared_device = f"fp-{rng.integers(10_000, 99_999)}"
            shared_phone = f"+91-{rng.integers(6000, 9999)}"

            for identity in range(int(params["identities_per_cluster"])):
                customer_id = f"A4-C{cluster:03d}-{identity:03d}"
                common = {
                    "customer_id": customer_id,
                    "card_id": f"A4-CARD-{cluster:03d}-{identity:03d}",
                    "card_bin": str(templates.iloc[rng.integers(len(templates))]["card_bin"]),
                    "device_prefix": (
                        shared_device if rng.random() < overlap else f"fp-{customer_id}"
                    ),
                    "address": (
                        shared_address if rng.random() < overlap else f"addr-{customer_id}"
                    ),
                    "phone_block": (
                        shared_phone if rng.random() < overlap else f"phone-{customer_id}"
                    ),
                    "cluster_id": cluster_id,
                    "playbook_id": playbook_id,
                }
                base_limit = float(rng.integers(35_000, 70_001))
                identity_exponent = max(
                    0.6,
                    float(params["utilization_curve_exponent"]) + rng.normal(0, variance * 0.8),
                )
                bust_day = (
                    build_days + 1 + int(rng.integers(0, int(params["bustout_sync_window_days"])))
                )
                schedule = np.arange(3, build_days + 1, 3)
                if variance:
                    schedule = np.unique(
                        np.clip(
                            schedule
                            + np.rint(rng.normal(0, variance * 4, len(schedule))).astype(int),
                            1,
                            build_days,
                        )
                    )

                rows.append(
                    self._row(
                        common,
                        event_number,
                        seed,
                        day=0,
                        phase="ONBOARD",
                        amount=0,
                        credit_limit=base_limit,
                        utilization=0,
                        payment_ratio=1,
                        template=None,
                        rng=rng,
                    )
                )
                event_number += 1
                seen_merchants: set[str] = set()

                for day in schedule:
                    progress = day / build_days
                    credit_limit = base_limit * (1 + 1.25 * progress)
                    utilization = np.clip(
                        0.08
                        + 0.38 * progress**identity_exponent
                        + rng.normal(0, 0.008 + variance * 0.04),
                        0.02,
                        0.62,
                    )
                    payment_ratio = np.clip(
                        1 - abs(rng.normal(0, 0.008 + variance * 0.12)), 0.65, 1
                    )
                    template = templates.iloc[rng.integers(len(templates))]
                    merchant = str(template["counterparty_id"])
                    amount = max(
                        1,
                        float(template["amount"]) * rng.lognormal(0, 0.12 + variance * 0.2),
                    )
                    row = self._row(
                        common,
                        event_number,
                        seed,
                        day=int(day),
                        phase="BUILD",
                        amount=amount,
                        credit_limit=credit_limit,
                        utilization=float(utilization),
                        payment_ratio=float(payment_ratio),
                        template=template,
                        rng=rng,
                    )
                    row["is_first_time_payee"] = int(merchant not in seen_merchants)
                    seen_merchants.add(merchant)
                    rows.append(row)
                    event_number += 1

                final_limit = base_limit * 2.25
                for burst, utilization in enumerate((0.52, 0.97)):
                    template = templates.iloc[rng.integers(len(templates))]
                    rows.append(
                        self._row(
                            common,
                            event_number,
                            seed,
                            day=bust_day,
                            phase="BUSTOUT",
                            amount=final_limit * (0.48 if burst == 0 else 0.45),
                            credit_limit=final_limit,
                            utilization=utilization,
                            payment_ratio=0,
                            template=template,
                            rng=rng,
                            hour=10 + burst * 6,
                        )
                    )
                    event_number += 1

        events = (
            coerce(pd.DataFrame(rows))
            .sort_values(["timestamp", "event_id"], kind="stable")
            .reset_index(drop=True)
        )
        validate(events)
        return events

    def utility(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        bustout = df["meta"].map(lambda meta: meta.get("a4_phase") == "BUSTOUT")
        extracted = float(df.loc[bustout & df["auth_result"].eq("APPROVED"), "amount"].sum())
        identities = max(1, df["customer_id"].nunique())
        build_days = max(
            1,
            int(df["meta"].map(lambda meta: meta.get("a4_days_since_onboarding", 0)).max()),
        )
        return float(np.tanh(extracted / (build_days * identities * 1_500)))

    def _validate_params(self, supplied: dict) -> dict:
        params = self.default_params() | supplied
        unknown = set(params).difference(self.param_space())
        if unknown:
            raise ValueError(f"unknown A4 parameters: {sorted(unknown)}")
        for name, (low, high) in self.param_space().items():
            value = params[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        integer_names = {
            "n_clusters",
            "identities_per_cluster",
            "build_days",
            "bustout_sync_window_days",
        }
        for name in integer_names:
            if not float(params[name]).is_integer():
                raise ValueError(f"{name} must be an integer")
            params[name] = int(params[name])
        return params

    @staticmethod
    def _row(
        common: dict[str, str],
        event_number: int,
        seed: int,
        *,
        day: int,
        phase: str,
        amount: float,
        credit_limit: float,
        utilization: float,
        payment_ratio: float,
        template: pd.Series | None,
        rng: np.random.Generator,
        hour: int | None = None,
    ) -> dict[str, Any]:
        day = int(day)
        selected_hour = hour if hour is not None else int(rng.integers(8, 22))
        timestamp = (
            pd.Timestamp("2026-01-01")
            + pd.to_timedelta(day, unit="D")
            + pd.to_timedelta(selected_hour, unit="h")
        )
        merchant_id = "KYC_ONBOARDING" if template is None else str(template["counterparty_id"])
        merchant_mcc = "0000" if template is None else str(template["counterparty_mcc"])
        merchant_age = 0 if template is None else int(template["counterparty_account_age_days"])
        return {
            "event_id": f"A4-{seed}-{event_number:08d}",
            "timestamp": timestamp,
            "attack_id": "A4",
            "label": 1,
            "round": 0,
            "customer_id": common["customer_id"],
            "customer_account_age_days": day + 1,
            "device_id": f"{common['device_prefix']}-{common['customer_id']}",
            "ip_asn": f"AS{int(rng.integers(1_000, 99_999))}",
            "session_id": f"A4S-{seed}-{event_number:08d}",
            "session_duration_s": int(rng.integers(45, 601)),
            "channel": "CNP" if rng.random() < 0.72 else "CARD_PRESENT",
            "card_id": common["card_id"],
            "card_bin": common["card_bin"],
            "vpa": "",
            "token_id": "",
            "counterparty_id": merchant_id,
            "counterparty_type": "MERCHANT",
            "counterparty_mcc": merchant_mcc,
            "counterparty_account_age_days": merchant_age,
            "amount": round(float(amount), 2),
            "is_first_time_payee": int(template is not None),
            "auth_result": "APPROVED",
            "decline_code": "",
            "meta": {
                "a4_identity_cluster_id": common["cluster_id"],
                "a4_playbook_id": common["playbook_id"],
                "a4_days_since_onboarding": day,
                "a4_credit_limit": round(float(credit_limit), 2),
                "a4_utilization": round(float(utilization), 4),
                "a4_payment_ratio": round(float(payment_ratio), 4),
                "a4_phase": phase,
                "a4_address_token": common["address"],
                "a4_device_prefix": common["device_prefix"],
                "a4_phone_block": common["phone_block"],
            },
        }
