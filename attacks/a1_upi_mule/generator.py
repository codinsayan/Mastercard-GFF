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
LAYER1_RETAINED_FRACTION = 0.02
LAYER2_RETAINED_FRACTION = 0.03
MAX_LAYER2_MULES = 10_000
MULE_ACQUISITION_COST = 10_000.0
CASHOUT_POINT_COST = 50_000.0
REFERENCE_PAYOFF = 0.25


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
        campaigns: list[dict] = []
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

            campaigns.append(
                {
                    "campaign_id": campaign_id,
                    "campaign_sequence": campaign_sequence,
                    "mule_number": mule_number,
                    "mule_id": mule_id,
                    "cashout_id": cashout_id,
                    "last_inflow_at": timestamp,
                    "received": round(float(sum(amounts)), 2),
                }
            )

        # Iteration 2: move each campaign through layer 1, layer 2, and a bounded
        # shared cash-out pool. Fractions retained at each hop model fees and keep
        # every node's outgoing amount strictly below its incoming amount.
        layer2_pool_size = min(
            MAX_LAYER2_MULES,
            checked["n_mules_layer1"] * checked["split_factor"],
        )
        for campaign in campaigns:
            layer1_total = round(
                campaign["received"] * (1.0 - LAYER1_RETAINED_FRACTION), 2
            )
            layer1_amounts = self._split_amount(
                layer1_total, checked["split_factor"], rng
            )
            layer1_departure = campaign["last_inflow_at"] + timedelta(
                hours=checked["dwell_time_hours"]
            )

            for branch, layer1_amount in enumerate(layer1_amounts):
                event_sequence += 1
                layer2_number = (
                    campaign["mule_number"] * checked["split_factor"] + branch
                ) % layer2_pool_size
                layer2_id = f"a1_mule_l2_{seed}_{layer2_number:05d}"
                layer1_timestamp = layer1_departure + timedelta(seconds=branch)
                cashout_number = layer2_number % checked["n_cashout_points"]
                cashout_id = f"a1_cashout_{seed}_{cashout_number:04d}"
                rows.append(
                    self._network_row(
                        event_id=f"a1_{seed}_{event_sequence:09d}",
                        timestamp=layer1_timestamp,
                        customer_id=campaign["mule_id"],
                        customer_age=checked["mule_age_days"],
                        counterparty_id=layer2_id,
                        counterparty_age=checked["mule_age_days"],
                        amount=layer1_amount,
                        seed=seed,
                        event_sequence=event_sequence,
                        campaign_id=campaign["campaign_id"],
                        cashout_id=cashout_id,
                        destination_layer=2,
                        source_role="MULE_L1",
                        destination_role="MULE_L2",
                    )
                )

                event_sequence += 1
                layer2_amount = round(
                    layer1_amount * (1.0 - LAYER2_RETAINED_FRACTION), 2
                )
                layer2_timestamp = layer1_timestamp + timedelta(
                    minutes=int(rng.integers(5, 31))
                )
                rows.append(
                    self._network_row(
                        event_id=f"a1_{seed}_{event_sequence:09d}",
                        timestamp=layer2_timestamp,
                        customer_id=layer2_id,
                        customer_age=checked["mule_age_days"],
                        counterparty_id=cashout_id,
                        counterparty_age=max(checked["mule_age_days"], 30),
                        amount=layer2_amount,
                        seed=seed,
                        event_sequence=event_sequence,
                        campaign_id=campaign["campaign_id"],
                        cashout_id=cashout_id,
                        destination_layer=3,
                        source_role="MULE_L2",
                        destination_role="CASHOUT",
                    )
                )

        result = coerce(pd.DataFrame(rows))
        result = result.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
        validate(result)
        return result

    def utility(self, df: pd.DataFrame) -> float:
        """Return normalized cash-out yield per unit of criminal infrastructure cost.

        Only complete, successfully approved A1 cash-out flows contribute yield. Mule
        cost rises with account age, making pre-aging an explicit economic trade-off.
        Malformed or unrelated rows are ignored instead of breaking the red-agent loop.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0
        required = {"amount", "counterparty_id", "counterparty_account_age_days", "meta"}
        if not required.issubset(df.columns):
            return 0.0

        layers = df["meta"].map(lambda value: self._meta_value(value, "a1_mule_layer"))
        amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if "auth_result" in df:
            approved = df["auth_result"].eq("APPROVED")
        else:
            approved = pd.Series(True, index=df.index)

        cashout_mask = layers.eq(3) & approved
        cashed_out_value = float(amounts.loc[cashout_mask].sum())
        if not np.isfinite(cashed_out_value) or cashed_out_value <= 0.0:
            return 0.0

        mule_rows = df.loc[layers.isin([1, 2]), [
            "counterparty_id",
            "counterparty_account_age_days",
        ]].copy()
        mule_rows = mule_rows.loc[mule_rows["counterparty_id"].astype("string").str.len().gt(0)]
        mule_rows["counterparty_account_age_days"] = pd.to_numeric(
            mule_rows["counterparty_account_age_days"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        unique_mules = mule_rows.drop_duplicates("counterparty_id")
        # Pre-aged accounts are more expensive to source and keep alive before use.
        age_multiplier = 1.0 + unique_mules["counterparty_account_age_days"] / 120.0
        mule_cost = float((MULE_ACQUISITION_COST * age_multiplier).sum())

        cashout_count = int(
            df.loc[cashout_mask, "counterparty_id"]
            .astype("string")
            .loc[lambda values: values.str.len().gt(0)]
            .nunique()
        )
        infrastructure_cost = mule_cost + cashout_count * CASHOUT_POINT_COST
        if not np.isfinite(infrastructure_cost) or infrastructure_cost <= 0.0:
            return 0.0

        raw_payoff = cashed_out_value / infrastructure_cost
        normalized = raw_payoff / (raw_payoff + REFERENCE_PAYOFF)
        return float(np.clip(normalized, 0.0, 1.0))

    def describe_state(self, params: dict) -> dict:
        self._validate_params(params)
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

        return self._validate_params(params)

    def _validate_params(self, params: dict) -> dict:
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
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
    def _meta_value(value, key: str):
        return value.get(key) if isinstance(value, dict) else None

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
        total_cents = round(total * 100)
        if total_cents < count:
            raise ValueError("total is too small to allocate at least one cent per transfer")
        if count == 1:
            return [total_cents / 100.0]
        remaining = total_cents - count
        weights = rng.dirichlet(np.full(count, 2.0))
        extra = np.floor(remaining * weights).astype(np.int64)
        extra[-1] += remaining - int(extra.sum())
        return [float(value + 1) / 100.0 for value in extra]

    @staticmethod
    def _network_row(
        *,
        event_id: str,
        timestamp,
        customer_id: str,
        customer_age: int,
        counterparty_id: str,
        counterparty_age: int,
        amount: float,
        seed: int,
        event_sequence: int,
        campaign_id: str,
        cashout_id: str,
        destination_layer: int,
        source_role: str,
        destination_role: str,
    ) -> dict:
        payee_added_at = timestamp - timedelta(minutes=10)
        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "attack_id": "A1",
            "label": 1,
            "round": 0,
            "customer_id": customer_id,
            "customer_account_age_days": customer_age,
            "device_id": f"a1_device_{customer_id}",
            "ip_asn": f"AS{50_000 + event_sequence % 10_000}",
            "session_id": f"a1_session_{seed}_{event_sequence:09d}",
            "session_duration_s": 60 + event_sequence % 240,
            "channel": "UPI_P2P",
            "card_id": "",
            "card_bin": "",
            "vpa": f"{customer_id}@bank",
            "token_id": "",
            "counterparty_id": counterparty_id,
            "counterparty_type": "BENEFICIARY",
            "counterparty_mcc": "",
            "counterparty_account_age_days": counterparty_age,
            "amount": amount,
            "is_first_time_payee": 1,
            "auth_result": "APPROVED",
            "decline_code": "",
            "meta": {
                "a1_call_active": False,
                "a1_payee_added_at": payee_added_at.isoformat(),
                "a1_mule_layer": destination_layer,
                "a1_cashout_node": cashout_id,
                "a1_campaign_id": campaign_id,
                "a1_source_role": source_role,
                "a1_destination_role": destination_role,
            },
        }
