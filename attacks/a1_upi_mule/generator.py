"""Iteration-1 generator for victim-to-layer-1 UPI coercion transfers."""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from core.schema import coerce, validate

CARD_PATH = Path(__file__).with_name("card.json")
INTEGER_PARAMS = {
    "n_victims",
    "mule_age_days",
    "split_factor",
    "n_cashout_points",
    "n_mules_layer1",
}
FLOAT_PARAMS = {"dwell_time_hours", "amount_multiplier_vs_baseline"}
POPULATION_COLUMNS = {
    "timestamp",
    "customer_id",
    "customer_account_age_days",
    "device_id",
    "ip_asn",
    "vpa",
    "session_duration_s",
    "amount",
    "auth_result",
}
MINIMUM_BASELINE_AMOUNT = 100.0


def _load_card() -> dict:
    return json.loads(CARD_PATH.read_text(encoding="utf-8"))


class A1Generator:
    """Generate deterministic victim-authorized transfers to layer-1 mule accounts."""

    attack_id = "A1"

    def default_params(self) -> dict:
        return copy.deepcopy(_load_card()["default_params"])

    def param_space(self) -> dict[str, tuple[float, float]]:
        return {
            name: (bounds[0], bounds[1])
            for name, bounds in _load_card()["param_space"].items()
        }

    def generate(self, params: dict, population: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Return victim-to-layer-1 fraud rows matching the frozen event schema."""
        checked = self._validate_inputs(params, population)
        history = self._customer_history(population)
        if history.empty:
            raise ValueError("population has no eligible approved customer history")

        rng = np.random.default_rng(seed)
        customer_ids = history.index.to_numpy(dtype=object)
        effective_victims = min(checked["n_victims"], len(customer_ids))
        selected = rng.choice(customer_ids, size=effective_victims, replace=False)
        campaign_start = pd.Timestamp(population["timestamp"].max()).to_pydatetime() + timedelta(hours=1)
        rows: list[dict] = []
        event_sequence = 0

        for campaign_sequence, customer_id in enumerate(selected):
            profile = history.loc[customer_id]
            campaign_id = f"a1_camp_{seed}_{campaign_sequence:06d}"
            mule_number = campaign_sequence % checked["n_mules_layer1"]
            mule_id = f"a1_mule_l1_{seed}_{mule_number:04d}"
            cashout_number = campaign_sequence % checked["n_cashout_points"]
            cashout_id = f"a1_cashout_{seed}_{cashout_number:04d}"
            payment_count = int(rng.integers(1, 4))
            baseline = max(float(profile["mean_amount"]), MINIMUM_BASELINE_AMOUNT)
            total = round(baseline * checked["amount_multiplier_vs_baseline"], 2)
            amounts = self._split_amount(total, payment_count, rng)
            campaign_offset = timedelta(seconds=int(rng.integers(0, 86_400)))
            payee_added_at = campaign_start + campaign_offset
            first_payment_delay = int(rng.integers(30, 301))
            prior_session = max(float(profile["median_session_duration_s"]), 1.0)

            for payment_number, amount in enumerate(amounts):
                event_sequence += 1
                timestamp = payee_added_at + timedelta(
                    seconds=first_payment_delay + payment_number * int(rng.integers(45, 181))
                )
                session_duration = int(
                    np.clip(prior_session * rng.uniform(1.5, 3.0), 60, np.iinfo(np.int32).max)
                )
                rows.append(
                    {
                        "event_id": f"a1_{seed}_{event_sequence:09d}",
                        "timestamp": timestamp,
                        "attack_id": "A1",
                        "label": 1,
                        "round": 0,
                        "customer_id": str(customer_id),
                        "customer_account_age_days": int(profile["customer_account_age_days"]),
                        "device_id": str(profile["device_id"]),
                        "ip_asn": str(profile["ip_asn"]),
                        "session_id": f"a1_session_{seed}_{event_sequence:09d}",
                        "session_duration_s": session_duration,
                        "channel": "UPI_P2P",
                        "card_id": "",
                        "card_bin": "",
                        "vpa": str(profile["vpa"]),
                        "token_id": "",
                        "counterparty_id": mule_id,
                        "counterparty_type": "BENEFICIARY",
                        "counterparty_mcc": "",
                        "counterparty_account_age_days": checked["mule_age_days"],
                        "amount": amount,
                        "is_first_time_payee": 1,
                        "auth_result": "APPROVED",
                        "decline_code": "",
                        "meta": {
                            "a1_call_active": True,
                            "a1_payee_added_at": payee_added_at.isoformat(),
                            "a1_mule_layer": 1,
                            "a1_cashout_node": cashout_id,
                            "a1_campaign_id": campaign_id,
                            "a1_source_role": "VICTIM",
                            "a1_destination_role": "MULE_L1",
                        },
                    }
                )

        result = coerce(pd.DataFrame(rows))
        result = result.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
        validate(result)
        return result

    def utility(self, df: pd.DataFrame) -> float:
        """Iteration-1 provisional yield score; full infrastructure cost arrives in Iteration 3."""
        if df.empty:
            return 0.0
        amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).clip(lower=0).sum()
        return float(amount / (amount + 1_000_000.0))

    def describe_state(self, params: dict) -> dict:
        card = _load_card()
        return {
            "attack_summary": card["attack_summary"],
            "param_semantics": copy.deepcopy(card["param_semantics"]),
            "defender_signals": copy.deepcopy(card["defender_signals"]),
            "current_params": copy.deepcopy(params),
        }

    def _validate_inputs(self, params: dict, population: pd.DataFrame) -> dict:
        if not isinstance(population, pd.DataFrame) or population.empty:
            raise ValueError("population must be a non-empty pandas DataFrame")
        missing_columns = POPULATION_COLUMNS.difference(population.columns)
        if missing_columns:
            raise ValueError(f"population is missing required columns: {sorted(missing_columns)}")
        if not pd.api.types.is_datetime64_ns_dtype(population["timestamp"]):
            raise TypeError("population timestamp must have dtype datetime64[ns]")

        expected = set(self.param_space())
        supplied = set(params)
        if supplied != expected:
            missing = sorted(expected.difference(supplied))
            unknown = sorted(supplied.difference(expected))
            raise ValueError(f"parameter keys do not match contract; missing={missing}, unknown={unknown}")

        checked: dict[str, int | float] = {}
        for name, value in params.items():
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be numeric, not boolean")
            if name in INTEGER_PARAMS:
                if not isinstance(value, Integral):
                    raise TypeError(f"{name} must be an integer")
                checked[name] = int(value)
            elif name in FLOAT_PARAMS:
                if not isinstance(value, Real):
                    raise TypeError(f"{name} must be numeric")
                checked[name] = float(value)
            else:
                raise ValueError(f"unsupported parameter: {name}")
            low, high = self.param_space()[name]
            if not low <= checked[name] <= high:
                raise ValueError(f"{name}={value} is outside [{low}, {high}]")
        return checked

    @staticmethod
    def _customer_history(population: pd.DataFrame) -> pd.DataFrame:
        approved = population.loc[
            population["auth_result"].eq("APPROVED")
            & pd.to_numeric(population["amount"], errors="coerce").gt(0)
        ].copy()
        if approved.empty:
            return pd.DataFrame()
        approved["amount"] = pd.to_numeric(approved["amount"], errors="coerce")
        approved = approved.sort_values(["timestamp", "customer_id"], kind="stable")
        amounts = approved.groupby("customer_id", sort=True)["amount"].agg(
            mean_amount="mean", median_amount="median"
        )
        sessions = approved.groupby("customer_id", sort=True)["session_duration_s"].median().rename(
            "median_session_duration_s"
        )
        latest = approved.groupby("customer_id", sort=True).tail(1).set_index("customer_id")
        profile_columns = [
            "customer_account_age_days",
            "device_id",
            "ip_asn",
            "vpa",
        ]
        return amounts.join(sessions).join(latest[profile_columns]).sort_index()

    @staticmethod
    def _split_amount(total: float, count: int, rng: np.random.Generator) -> list[float]:
        if count == 1:
            return [round(total, 2)]
        weights = rng.dirichlet(np.full(count, 2.0))
        amounts = np.round(total * weights, 2)
        amounts[-1] = round(total - float(amounts[:-1].sum()), 2)
        return [float(value) for value in amounts]
