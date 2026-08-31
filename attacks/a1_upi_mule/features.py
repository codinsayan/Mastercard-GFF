"""Round-0 A1 features for coerced UPI payments and mule-account behavior."""

from __future__ import annotations

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

REQUIRED_INPUTS = {
    "event_id",
    "timestamp",
    "customer_id",
    "customer_account_age_days",
    "session_duration_s",
    "counterparty_id",
    "counterparty_account_age_days",
    "amount",
    "meta",
}
DAY_NS = 86_400_000_000_000
NINETY_DAYS_NS = 90 * DAY_NS
GRAPH_WINDOW_NS = 172_800_000_000_000


class A1Features:
    """Build leakage-aware node-level features for the A1 Round-0 detector."""

    attack_id = "A1"

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        """Return five numeric A1 features indexed in the original event order.

        Fan-in and payer statistics use strictly prior transactions. Dissipation is
        intentionally a delayed signal: it observes beneficiary outflow during the hour
        after the scored transfer and therefore supports near-real-time mule monitoring,
        not authorization-time scoring of the first victim payment.
        """
        self._validate(events)
        if events.empty:
            index = pd.Index([], dtype="string", name="event_id")
            return pd.DataFrame(columns=self.feature_names(), index=index, dtype="float64")

        inputs = events.loc[:, sorted(REQUIRED_INPUTS.difference({"meta"}))].copy()
        inputs["_input_order"] = np.arange(len(inputs), dtype=np.int64)
        query = f"""
            SELECT
                current._input_order,
                current.event_id,
                CAST(current.counterparty_account_age_days AS DOUBLE)
                    AS a1_benef_account_age_days,
                CAST(
                    count(DISTINCT current.customer_id) OVER (
                        PARTITION BY current.counterparty_id
                        ORDER BY epoch_ns(current.timestamp)
                        RANGE BETWEEN {DAY_NS} PRECEDING AND 1 PRECEDING
                    ) AS DOUBLE
                ) AS a1_benef_fanin_24h,
                CAST(
                    CASE
                        WHEN payer_std IS NULL OR payer_std = 0 THEN 0
                        ELSE (current.amount - payer_mean) / payer_std
                    END AS DOUBLE
                ) AS a1_payer_amount_zscore,
                CAST(
                    CASE
                        WHEN session_std IS NULL OR session_std = 0 THEN 0
                        ELSE (current.session_duration_s - session_mean) / session_std
                    END AS DOUBLE
                ) AS a1_session_duration_anomaly,
                CAST(coalesce(outflow.outgoing_1h, 0) / nullif(current.amount, 0) AS DOUBLE)
                    AS a1_dissipation_ratio
            FROM (
                SELECT
                    *,
                    avg(amount) OVER (
                        PARTITION BY customer_id
                        ORDER BY epoch_ns(timestamp)
                        RANGE BETWEEN {NINETY_DAYS_NS} PRECEDING AND 1 PRECEDING
                    ) AS payer_mean,
                    stddev_samp(amount) OVER (
                        PARTITION BY customer_id
                        ORDER BY epoch_ns(timestamp)
                        RANGE BETWEEN {NINETY_DAYS_NS} PRECEDING AND 1 PRECEDING
                    ) AS payer_std,
                    avg(session_duration_s) OVER (
                        PARTITION BY customer_id
                        ORDER BY epoch_ns(timestamp)
                        RANGE BETWEEN {NINETY_DAYS_NS} PRECEDING AND 1 PRECEDING
                    ) AS session_mean,
                    stddev_samp(session_duration_s) OVER (
                        PARTITION BY customer_id
                        ORDER BY epoch_ns(timestamp)
                        RANGE BETWEEN {NINETY_DAYS_NS} PRECEDING AND 1 PRECEDING
                    ) AS session_std
                FROM events_input
            ) AS current
            LEFT JOIN LATERAL (
                SELECT sum(future.amount) AS outgoing_1h
                FROM events_input AS future
                WHERE future.customer_id = current.counterparty_id
                  AND future.timestamp > current.timestamp
                  AND future.timestamp <= current.timestamp + INTERVAL 1 HOUR
            ) AS outflow ON true
            ORDER BY current._input_order
        """
        with duckdb.connect() as connection:
            connection.register("events_input", inputs)
            output = connection.execute(query).df()

        output["a1_payee_added_to_transfer_seconds"] = self._payee_delay(events)
        output["a1_call_active"] = events["meta"].map(
            lambda value: float(bool(value.get("a1_call_active", False)))
            if isinstance(value, dict)
            else 0.0
        ).to_numpy()
        graph = self._graph_features(events)
        for name in graph:
            output[name] = graph[name]
        output = output.drop(columns="_input_order").set_index("event_id")
        output.index = output.index.astype("string")
        output.index.name = "event_id"
        output = output.loc[:, self.feature_names()]
        return output.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "a1_benef_account_age_days",
            "a1_benef_fanin_24h",
            "a1_dissipation_ratio",
            "a1_payer_amount_zscore",
            "a1_payee_added_to_transfer_seconds",
            "a1_component_size",
            "a1_component_dissipation",
            "a1_cashout_convergence",
            "a1_component_age_variance",
            "a1_call_active",
            "a1_session_duration_anomaly",
        ]

    @staticmethod
    def _validate(events: pd.DataFrame) -> None:
        if not isinstance(events, pd.DataFrame):
            raise TypeError("events must be a pandas DataFrame")
        missing = REQUIRED_INPUTS.difference(events.columns)
        if missing:
            raise ValueError(f"events are missing required columns: {sorted(missing)}")
        if events["event_id"].duplicated().any():
            raise ValueError("event_id must be unique")
        if not pd.api.types.is_datetime64_ns_dtype(events["timestamp"]):
            raise TypeError("timestamp must have dtype datetime64[ns]")
        amount = pd.to_numeric(events["amount"], errors="coerce")
        if amount.isna().any() or (amount <= 0).any():
            raise ValueError("amount must contain finite positive numeric values")

    @staticmethod
    def _payee_delay(events: pd.DataFrame) -> np.ndarray:
        raw = events["meta"].map(
            lambda value: value.get("a1_payee_added_at") if isinstance(value, dict) else None
        )
        added = pd.to_datetime(raw, errors="coerce")
        delay = (events["timestamp"].reset_index(drop=True) - added).dt.total_seconds()
        # Missing, malformed, negative, and infinite values use the documented neutral 0.
        return delay.where(delay.ge(0) & np.isfinite(delay), 0.0).fillna(0.0).to_numpy()

    @classmethod
    def _graph_features(cls, events: pd.DataFrame) -> dict[str, np.ndarray]:
        """Compute delayed 48-hour local topology without building a global graph per row."""
        sources = events["customer_id"].astype(str).to_numpy()
        targets = events["counterparty_id"].astype(str).to_numpy()
        timestamps = events["timestamp"].to_numpy(dtype="datetime64[ns]")
        amounts = pd.to_numeric(events["amount"], errors="coerce").to_numpy(dtype=float)
        source_ages = pd.to_numeric(
            events.get("customer_account_age_days", pd.Series(0, index=events.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        target_ages = pd.to_numeric(events["counterparty_account_age_days"], errors="coerce").fillna(
            0.0
        ).to_numpy(dtype=float)

        incident: dict[str, list[int]] = {}
        for edge_index, (source, target) in enumerate(zip(sources, targets, strict=True)):
            incident.setdefault(source, []).append(edge_index)
            incident.setdefault(target, []).append(edge_index)

        size = np.zeros(len(events), dtype=float)
        dissipation = np.zeros(len(events), dtype=float)
        convergence = np.zeros(len(events), dtype=float)
        age_variance = np.zeros(len(events), dtype=float)
        window_ns = GRAPH_WINDOW_NS

        for row in range(len(events)):
            beneficiary = targets[row]
            center = timestamps[row].astype("datetime64[ns]").astype(np.int64)
            local_edges = cls._local_edge_indices(
                beneficiary, center, window_ns, incident, sources, targets, timestamps
            )
            if not local_edges:
                continue
            graph = nx.DiGraph()
            ages: dict[str, float] = {}
            for edge_index in local_edges:
                source, target = sources[edge_index], targets[edge_index]
                amount = float(amounts[edge_index])
                if graph.has_edge(source, target):
                    graph[source][target]["amount"] += amount
                else:
                    graph.add_edge(source, target, amount=amount)
                ages.setdefault(source, float(source_ages[edge_index]))
                ages.setdefault(target, float(target_ages[edge_index]))

            if beneficiary not in graph:
                continue
            undirected = graph.to_undirected()
            distances = nx.single_source_shortest_path_length(undirected, beneficiary, cutoff=2)
            component = set(distances)
            size[row] = float(len(component))
            known_ages = [ages[node] for node in component if ages.get(node, 0.0) >= 0.0]
            age_variance[row] = float(np.var(known_ages, ddof=1)) if len(known_ages) >= 2 else 0.0

            component_in = 0.0
            component_out = 0.0
            for source, target, data in graph.edges(data=True):
                if target in component:
                    component_in += float(data["amount"])
                if source in component:
                    component_out += float(data["amount"])
            dissipation[row] = component_out / component_in if component_in > 0 else 0.0
            convergence[row] = cls._cashout_convergence(graph, beneficiary)

        return {
            "a1_component_size": size,
            "a1_component_dissipation": dissipation,
            "a1_cashout_convergence": convergence,
            "a1_component_age_variance": age_variance,
        }

    @staticmethod
    def _local_edge_indices(
        beneficiary: str,
        center_ns: int,
        window_ns: int,
        incident: dict[str, list[int]],
        sources: np.ndarray,
        targets: np.ndarray,
        timestamps: np.ndarray,
    ) -> set[int]:
        """Collect a three-hop neighborhood inside ±48h for delayed graph monitoring."""
        frontier = {beneficiary}
        visited = {beneficiary}
        selected: set[int] = set()
        for _ in range(4):
            next_frontier: set[str] = set()
            for node in frontier:
                for edge_index in incident.get(node, []):
                    edge_ns = timestamps[edge_index].astype("datetime64[ns]").astype(np.int64)
                    if abs(int(edge_ns) - center_ns) > window_ns:
                        continue
                    selected.add(edge_index)
                    for adjacent in (sources[edge_index], targets[edge_index]):
                        if adjacent not in visited:
                            visited.add(adjacent)
                            next_frontier.add(adjacent)
            frontier = next_frontier
            if not frontier:
                break
        return selected

    @staticmethod
    def _cashout_convergence(graph: nx.DiGraph, beneficiary: str) -> float:
        """Maximum distinct two-hop upstream mule count at reachable terminal nodes."""
        reachable = nx.descendants(graph, beneficiary) | {beneficiary}
        sinks = [node for node in reachable if graph.out_degree(node) == 0]
        maximum = 0
        reverse = graph.reverse(copy=False)
        for sink in sinks:
            upstream = nx.single_source_shortest_path_length(reverse, sink, cutoff=2)
            layer1_sources = {node for node, distance in upstream.items() if distance == 2}
            maximum = max(maximum, len(layer1_sources))
        return float(maximum)
