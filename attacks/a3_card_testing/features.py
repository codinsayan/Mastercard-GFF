"""Causal round-0 and round-1 features for A3 agentic card testing."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

import networkx as nx
import numpy as np
import pandas as pd

ONE_HOUR_NS = 3_600_000_000_000
MICRO_TRANSACTION_MAX = 100.0
RETRY_WINDOW_NS = 300_000_000_000
RETRY_AMOUNT_RELATIVE_DELTA_MAX = 0.10
ROUND0_FEATURES = (
    "a3_device_card_fanin_1h",
    "a3_merchant_micro_txn_rate_1h",
    "a3_amount_percentile",
    "a3_card_first_seen_recency",
)
ROUND1_FEATURES = (
    "a3_merchant_set_jaccard",
    "a3_merchant_friction_score",
    "a3_bin_cooccurrence_score",
    "a3_bin_entropy_1h",
    "a3_decline_retry_pattern",
)
FEATURES = ROUND0_FEATURES + ROUND1_FEATURES


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
    """Build all nine causal A3 signals for every input event."""

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
            "card_bin",
            "auth_result",
            "meta",
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
        inputs = inputs.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
        return self._build_sorted(inputs)

    @staticmethod
    def _build_sorted(inputs: pd.DataFrame) -> pd.DataFrame:
        row_count = len(inputs)
        fanin = np.zeros(row_count, dtype=float)
        micro_rate = np.zeros(row_count, dtype=float)
        amount_percentile = np.full(row_count, 0.5, dtype=float)
        first_seen_recency = np.zeros(row_count, dtype=float)
        merchant_set_jaccard = np.zeros(row_count, dtype=float)
        merchant_friction_score = np.full(row_count, 0.5, dtype=float)
        bin_cooccurrence_score = np.zeros(row_count, dtype=float)
        bin_entropy = np.ones(row_count, dtype=float)
        decline_retry_pattern = np.zeros(row_count, dtype=float)
        if not row_count:
            return A3Features._output(
                inputs,
                fanin,
                micro_rate,
                amount_percentile,
                first_seen_recency,
                merchant_set_jaccard,
                merchant_friction_score,
                bin_cooccurrence_score,
                bin_entropy,
                decline_retry_pattern,
            )

        timestamps = inputs["timestamp"].astype("int64").to_numpy()
        devices = inputs["device_id"].fillna("").astype(str).to_numpy()
        cards = inputs["card_id"].fillna("").astype(str).to_numpy()
        merchants = inputs["counterparty_id"].fillna("").astype(str).to_numpy()
        counterparty_types = inputs["counterparty_type"].fillna("").astype(str).to_numpy()
        channels = inputs["channel"].fillna("").astype(str).to_numpy()
        amounts = pd.to_numeric(inputs["amount"], errors="coerce").to_numpy(dtype=float)
        card_bins = inputs["card_bin"].fillna("").astype(str).to_numpy()
        auth_results = inputs["auth_result"].fillna("").astype(str).to_numpy()
        metas = inputs["meta"].to_numpy()

        finite_amounts = amounts[np.isfinite(amounts)]
        amount_values = np.unique(finite_amounts)
        amount_tree = _FenwickTree(len(amount_values))
        prior_cnp_count = 0
        first_card_timestamp: dict[str, int] = {}

        device_events: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
        device_card_counts: dict[str, dict[str, int]] = defaultdict(dict)
        merchant_micro_events: dict[str, deque[int]] = defaultdict(deque)

        # A rolling NetworkX bipartite card <-> merchant graph. Counts retain an
        # edge until its final event leaves the one-hour window.
        merchant_graph = nx.Graph()
        merchant_edges: deque[tuple[int, str, str]] = deque()
        merchant_edge_counts: Counter[tuple[str, str]] = Counter()
        card_friction_events: dict[str, deque[tuple[int, float]]] = defaultdict(deque)

        active_card_events: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
        active_events: deque[tuple[int, str, str]] = deque()
        active_bin_counts: Counter[str] = Counter()
        background_card_bins: dict[str, str] = {}
        background_bin_counts: Counter[str] = Counter()
        prior_distinct_cards = 0
        declined_attempts: dict[str, deque[tuple[int, float]]] = defaultdict(deque)

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
                card_bin = card_bins[index]

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

                cutoff = timestamp - ONE_HOUR_NS
                while merchant_edges and merchant_edges[0][0] <= cutoff:
                    _, expired_card, expired_merchant = merchant_edges.popleft()
                    edge = (expired_card, expired_merchant)
                    merchant_edge_counts[edge] -= 1
                    if not merchant_edge_counts[edge]:
                        del merchant_edge_counts[edge]
                        card_node = ("card", expired_card)
                        merchant_node = ("merchant", expired_merchant)
                        if merchant_graph.has_edge(card_node, merchant_node):
                            merchant_graph.remove_edge(card_node, merchant_node)
                        for node in (card_node, merchant_node):
                            if node in merchant_graph and not merchant_graph.degree(node):
                                merchant_graph.remove_node(node)

                if is_merchant and card:
                    edge = (card, merchant)
                    merchant_edges.append((timestamp, card, merchant))
                    merchant_edge_counts[edge] += 1
                    merchant_graph.add_edge(("card", card), ("merchant", merchant))
                    merchant_set_jaccard[index] = A3Features._maximum_card_jaccard(
                        merchant_graph, card
                    )

                    friction = A3Features._friction_value(metas[index])
                    friction_queue = card_friction_events[card]
                    while friction_queue and friction_queue[0][0] <= cutoff:
                        friction_queue.popleft()
                    if friction is not None:
                        friction_queue.append((timestamp, friction))
                    if friction_queue:
                        merchant_friction_score[index] = float(
                            np.mean([value for _, value in friction_queue])
                        )

                A3Features._expire_active_cards(
                    active_events, active_card_events, active_bin_counts, cutoff
                )
                if card and card_bin:
                    previous_active_bin = (
                        active_card_events[card][-1][1] if active_card_events[card] else ""
                    )
                    if previous_active_bin:
                        active_bin_counts[previous_active_bin] -= 1
                        if not active_bin_counts[previous_active_bin]:
                            del active_bin_counts[previous_active_bin]
                    active_card_events[card].append((timestamp, card_bin))
                    active_events.append((timestamp, card, card_bin))
                    active_bin_counts[card_bin] += 1

                    active_count = sum(active_bin_counts.values())
                    local_share = active_bin_counts[card_bin] / active_count
                    background_share = (
                        background_bin_counts[card_bin] / prior_distinct_cards
                        if prior_distinct_cards
                        else 0.0
                    )
                    bin_cooccurrence_score[index] = max(0.0, local_share - background_share)
                    bin_entropy[index] = A3Features._normalised_entropy(active_bin_counts)

                if card:
                    retry_queue = declined_attempts[card]
                    retry_cutoff = timestamp - RETRY_WINDOW_NS
                    while retry_queue and retry_queue[0][0] <= retry_cutoff:
                        retry_queue.popleft()
                    if np.isfinite(amount):
                        decline_retry_pattern[index] = float(
                            any(
                                A3Features._is_perturbed_retry(prior_amount, amount)
                                for _, prior_amount in retry_queue
                            )
                        )
                    if auth_results[index] == "DECLINED" and np.isfinite(amount):
                        retry_queue.append((timestamp, amount))

            for index in range(group_start, group_end):
                amount = amounts[index]
                if channels[index] == "CNP" and np.isfinite(amount):
                    position = int(np.searchsorted(amount_values, amount, side="left"))
                    amount_tree.add(position)
                    prior_cnp_count += 1
                card = cards[index]
                if card:
                    first_card_timestamp.setdefault(card, timestamp)
                    card_bin = card_bins[index]
                    if card not in background_card_bins and card_bin:
                        background_card_bins[card] = card_bin
                        background_bin_counts[card_bin] += 1
                        prior_distinct_cards += 1

            group_start = group_end

        return A3Features._output(
            inputs,
            fanin,
            micro_rate,
            amount_percentile,
            first_seen_recency,
            merchant_set_jaccard,
            merchant_friction_score,
            bin_cooccurrence_score,
            bin_entropy,
            decline_retry_pattern,
        )

    @staticmethod
    def _maximum_card_jaccard(graph: nx.Graph, card: str) -> float:
        card_node = ("card", card)
        if card_node not in graph:
            return 0.0
        merchants = set(graph.neighbors(card_node))
        maximum = 0.0
        for other_node in graph:
            if other_node == card_node or other_node[0] != "card":
                continue
            other_merchants = set(graph.neighbors(other_node))
            union = merchants | other_merchants
            if union:
                maximum = max(maximum, len(merchants & other_merchants) / len(union))
        return maximum

    @staticmethod
    def _friction_value(meta: object) -> float | None:
        if not isinstance(meta, dict):
            return None
        tier = str(meta.get("a3_merchant_friction_tier", "")).upper()
        return {"LOW": 0.0, "MED": 0.5, "HIGH": 1.0}.get(tier)

    @staticmethod
    def _expire_active_cards(
        all_events: deque[tuple[int, str, str]],
        active_events: dict[str, deque[tuple[int, str]]],
        bin_counts: Counter[str],
        cutoff: int,
    ) -> None:
        while all_events and all_events[0][0] <= cutoff:
            timestamp, card, card_bin = all_events.popleft()
            events = active_events[card]
            previous_bin = events[-1][1] if events else ""
            if events and events[0] == (timestamp, card_bin):
                events.popleft()
            current_bin = events[-1][1] if events else ""
            if current_bin != previous_bin:
                bin_counts[previous_bin] -= 1
                if not bin_counts[previous_bin]:
                    del bin_counts[previous_bin]
                if current_bin:
                    bin_counts[current_bin] += 1
            if not events:
                del active_events[card]

    @staticmethod
    def _normalised_entropy(bin_counts: Counter[str]) -> float:
        counts = np.asarray(list(bin_counts.values()), dtype=float)
        if counts.size <= 1:
            return 0.0
        probabilities = counts / counts.sum()
        return float(-(probabilities * np.log(probabilities)).sum() / np.log(counts.size))

    @staticmethod
    def _is_perturbed_retry(prior_amount: float, amount: float) -> bool:
        if prior_amount <= 0.0 or amount == prior_amount:
            return False
        return abs(amount - prior_amount) / prior_amount <= RETRY_AMOUNT_RELATIVE_DELTA_MAX

    @staticmethod
    def _output(
        inputs: pd.DataFrame,
        fanin: np.ndarray,
        micro_rate: np.ndarray,
        amount_percentile: np.ndarray,
        first_seen_recency: np.ndarray,
        merchant_set_jaccard: np.ndarray,
        merchant_friction_score: np.ndarray,
        bin_cooccurrence_score: np.ndarray,
        bin_entropy: np.ndarray,
        decline_retry_pattern: np.ndarray,
    ) -> pd.DataFrame:
        output = pd.DataFrame(
            {
                "_input_order": inputs["_input_order"].to_numpy(),
                "event_id": inputs["event_id"].astype("string").to_numpy(),
                "a3_device_card_fanin_1h": fanin,
                "a3_merchant_micro_txn_rate_1h": micro_rate,
                "a3_amount_percentile": amount_percentile,
                "a3_card_first_seen_recency": first_seen_recency,
                "a3_merchant_set_jaccard": merchant_set_jaccard,
                "a3_merchant_friction_score": merchant_friction_score,
                "a3_bin_cooccurrence_score": bin_cooccurrence_score,
                "a3_bin_entropy_1h": bin_entropy,
                "a3_decline_retry_pattern": decline_retry_pattern,
            }
        )
        output = output.sort_values("_input_order", kind="stable").drop(columns="_input_order")
        output = output.set_index("event_id")
        output.index = output.index.astype("string")
        output.index.name = "event_id"
        return output.loc[:, list(FEATURES)]
