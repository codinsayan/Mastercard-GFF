"""Causal round-0 features for A3 agentic card testing."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

ONE_HOUR_NS = 3_600_000_000_000
MICRO_TRANSACTION_MAX = 100.0
ROUND0_FEATURES = (
    "a3_device_card_fanin_1h",
    "a3_merchant_micro_txn_rate_1h",
    "a3_amount_percentile",
    "a3_card_first_seen_recency",
)


class _FenwickTree:
    """Prefix-count tree used for an exact expanding empirical CDF."""

    def __init__(self, size: int):
        self.values = np.zeros(size + 1, dtype=np.int64)

    def add(self, position: int) -> None:
        index = position + 1
        while index < len(self.values):
            self.values[index] += 1
            index += index & -index

    def prefix_count(self, end: int) -> int:
        total = 0
        index = end
        while index:
            total += int(self.values[index])
            index -= index & -index
        return total


class A3Features:
    """Build the four causal A3 round-0 signals for every input event."""

    attack_id = "A3"

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        required = {
            "event_id",
            "timestamp",
            "device_id",
            "card_id",
            "counterparty_id",
            "counterparty_type",
            "amount",
            "channel",
        }
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"Missing A3 feature inputs: {sorted(missing)}")
        if events["event_id"].duplicated().any():
            raise ValueError("event_id must be unique")
        if events["timestamp"].dtype != np.dtype("datetime64[ns]"):
            raise TypeError("timestamp must have dtype datetime64[ns]")
        if events["timestamp"].isna().any():
            raise ValueError("timestamp cannot contain missing values")

        inputs = events.loc[:, sorted(required)].copy()
        inputs["_input_order"] = np.arange(len(inputs), dtype=np.int64)
        inputs = inputs.sort_values(["timestamp", "event_id"], kind="stable").reset_index(
            drop=True
        )
        return self._build_sorted(inputs)

    @staticmethod
    def _build_sorted(inputs: pd.DataFrame) -> pd.DataFrame:
        row_count = len(inputs)
        fanin = np.zeros(row_count, dtype=float)
        micro_rate = np.zeros(row_count, dtype=float)
        amount_percentile = np.full(row_count, 0.5, dtype=float)
        first_seen_recency = np.zeros(row_count, dtype=float)
        if not row_count:
            return A3Features._output(
                inputs, fanin, micro_rate, amount_percentile, first_seen_recency
            )

        timestamps = inputs["timestamp"].astype("int64").to_numpy()
        devices = inputs["device_id"].fillna("").astype(str).to_numpy()
        cards = inputs["card_id"].fillna("").astype(str).to_numpy()
        merchants = inputs["counterparty_id"].fillna("").astype(str).to_numpy()
        counterparty_types = inputs["counterparty_type"].fillna("").astype(str).to_numpy()
        channels = inputs["channel"].fillna("").astype(str).to_numpy()
        amounts = pd.to_numeric(inputs["amount"], errors="coerce").to_numpy(dtype=float)

        finite_amounts = amounts[np.isfinite(amounts)]
        amount_values = np.unique(finite_amounts)
        amount_tree = _FenwickTree(len(amount_values))
        prior_cnp_count = 0
        first_card_timestamp: dict[str, int] = {}

        device_events: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
        device_card_counts: dict[str, dict[str, int]] = defaultdict(dict)
        merchant_micro_events: dict[str, deque[int]] = defaultdict(deque)

        group_start = 0
        while group_start < row_count:
            timestamp = int(timestamps[group_start])
            group_end = group_start + 1
            while group_end < row_count and timestamps[group_end] == timestamp:
                group_end += 1

            for index in range(group_start, group_end):
                card = cards[index]
                device = devices[index]
                merchant = merchants[index]
                amount = amounts[index]

                if device and card:
                    queue = device_events[device]
                    counts = device_card_counts[device]
                    cutoff = timestamp - ONE_HOUR_NS
                    while queue and queue[0][0] <= cutoff:
                        _, expired_card = queue.popleft()
                        remaining = counts[expired_card] - 1
                        if remaining:
                            counts[expired_card] = remaining
                        else:
                            del counts[expired_card]
                    queue.append((timestamp, card))
                    counts[card] = counts.get(card, 0) + 1
                    fanin[index] = float(len(counts))

                is_merchant = counterparty_types[index] == "MERCHANT" and bool(merchant)
                if is_merchant and np.isfinite(amount):
                    queue = merchant_micro_events[merchant]
                    cutoff = timestamp - ONE_HOUR_NS
                    while queue and queue[0] <= cutoff:
                        queue.popleft()
                    if amount <= MICRO_TRANSACTION_MAX:
                        queue.append(timestamp)
                    micro_rate[index] = float(len(queue))

                if np.isfinite(amount) and prior_cnp_count:
                    end = int(np.searchsorted(amount_values, amount, side="right"))
                    amount_percentile[index] = amount_tree.prefix_count(end) / prior_cnp_count

                if card and card in first_card_timestamp:
                    elapsed_ns = timestamp - first_card_timestamp[card]
                    first_seen_recency[index] = elapsed_ns / 1_000_000_000

            for index in range(group_start, group_end):
                amount = amounts[index]
                if channels[index] == "CNP" and np.isfinite(amount):
                    position = int(np.searchsorted(amount_values, amount, side="left"))
                    amount_tree.add(position)
                    prior_cnp_count += 1
                card = cards[index]
                if card:
                    first_card_timestamp.setdefault(card, timestamp)

            group_start = group_end

        return A3Features._output(
            inputs, fanin, micro_rate, amount_percentile, first_seen_recency
        )

    @staticmethod
    def _output(
        inputs: pd.DataFrame,
        fanin: np.ndarray,
        micro_rate: np.ndarray,
        amount_percentile: np.ndarray,
        first_seen_recency: np.ndarray,
    ) -> pd.DataFrame:
        output = pd.DataFrame(
            {
                "_input_order": inputs["_input_order"].to_numpy(),
                "event_id": inputs["event_id"].astype("string").to_numpy(),
                "a3_device_card_fanin_1h": fanin,
                "a3_merchant_micro_txn_rate_1h": micro_rate,
                "a3_amount_percentile": amount_percentile,
                "a3_card_first_seen_recency": first_seen_recency,
            }
        )
        output = output.sort_values("_input_order", kind="stable").drop(columns="_input_order")
        output = output.set_index("event_id")
        output.index = output.index.astype("string")
        output.index.name = "event_id"
        return output.loc[:, list(ROUND0_FEATURES)]
