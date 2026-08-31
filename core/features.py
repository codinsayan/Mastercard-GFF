"""Shared leakage-safe temporal features available to every attack."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

WINDOW_NS = {
    "1h": 3_600_000_000_000,
    "24h": 86_400_000_000_000,
    "7d": 604_800_000_000_000,
    "30d": 2_592_000_000_000_000,
}


def _window_expressions() -> list[str]:
    expressions: list[str] = []
    for entity, prefix in (("customer_id", "customer"), ("counterparty_id", "counterparty")):
        for label, nanoseconds in WINDOW_NS.items():
            window = (
                f"PARTITION BY {entity} ORDER BY epoch_ns(timestamp) "
                f"RANGE BETWEEN {nanoseconds} PRECEDING AND 1 PRECEDING"
            )
            expressions.extend(
                [
                    f"count(amount) OVER ({window}) AS {prefix}_count_{label}",
                    f"sum(amount) OVER ({window}) AS {prefix}_sum_{label}",
                    f"avg(amount) OVER ({window}) AS {prefix}_mean_{label}",
                    f"stddev_samp(amount) OVER ({window}) AS {prefix}_std_{label}",
                ]
            )

    for label, nanoseconds in WINDOW_NS.items():
        customer_window = (
            "PARTITION BY customer_id ORDER BY epoch_ns(timestamp) "
            f"RANGE BETWEEN {nanoseconds} PRECEDING AND 1 PRECEDING"
        )
        counterparty_window = (
            "PARTITION BY counterparty_id ORDER BY epoch_ns(timestamp) "
            f"RANGE BETWEEN {nanoseconds} PRECEDING AND 1 PRECEDING"
        )
        expressions.extend(
            [
                (
                    "count(DISTINCT counterparty_id) "
                    f"OVER ({customer_window}) AS customer_distinct_counterparties_{label}"
                ),
                (
                    "count(DISTINCT device_id) "
                    f"OVER ({customer_window}) AS customer_distinct_devices_{label}"
                ),
                (
                    "count(DISTINCT customer_id) "
                    f"OVER ({counterparty_window}) AS counterparty_distinct_customers_{label}"
                ),
                (
                    "count(DISTINCT device_id) "
                    f"OVER ({counterparty_window}) AS counterparty_distinct_devices_{label}"
                ),
            ]
        )
    return expressions


def build_core_features(events: pd.DataFrame) -> pd.DataFrame:
    """Return prior-only rolling features indexed in the input event order."""
    required = {
        "event_id",
        "timestamp",
        "customer_id",
        "counterparty_id",
        "amount",
        "device_id",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing feature inputs: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    if not pd.api.types.is_datetime64_ns_dtype(events["timestamp"]):
        raise TypeError("timestamp must have dtype datetime64[ns]")

    inputs = events.loc[:, sorted(required)].copy()
    inputs["_input_order"] = np.arange(len(inputs), dtype=np.int64)
    prior_30d = (
        "PARTITION BY customer_id ORDER BY epoch_ns(timestamp) "
        f"RANGE BETWEEN {WINDOW_NS['30d']} PRECEDING AND 1 PRECEDING"
    )
    expressions = ",\n                ".join(_window_expressions())
    query = f"""
        SELECT
            _input_order,
            event_id,
            amount,
            CAST(hour(timestamp) AS TINYINT) AS hour_of_day,
            CAST(isodow(timestamp) - 1 AS TINYINT) AS day_of_week,
            coalesce(
                epoch(timestamp - lag(timestamp) OVER (
                    PARTITION BY customer_id ORDER BY timestamp, event_id
                )),
                -1
            ) AS time_since_previous_s,
            {expressions},
            avg(amount) OVER ({prior_30d}) AS _customer_mean_30d,
            stddev_samp(amount) OVER ({prior_30d}) AS _customer_std_30d
        FROM events_input
        ORDER BY _input_order
    """
    with duckdb.connect() as connection:
        connection.register("events_input", inputs)
        output = connection.execute(query).df()

    std = output.pop("_customer_std_30d").replace(0, np.nan)
    mean = output.pop("_customer_mean_30d")
    amount = output.pop("amount")
    output["amount_zscore_30d"] = (amount - mean) / std
    output = output.drop(columns="_input_order").set_index("event_id")
    output.index = output.index.astype("string")
    output.index.name = "event_id"
    return output.replace([np.inf, -np.inf], np.nan).fillna(0.0)
