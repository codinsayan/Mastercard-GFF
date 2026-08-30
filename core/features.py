"""Shared temporal features available to every attack."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"1h": "1h", "24h": "24h", "7d": "7d", "30d": "30d"}


def build_core_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe rolling features, indexed by event_id."""
    required = {"event_id", "timestamp", "customer_id", "counterparty_id", "amount", "device_id"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing feature inputs: {sorted(missing)}")
    ordered = events.sort_values(["timestamp", "event_id"], kind="stable").copy()
    ordered["hour_of_day"] = ordered["timestamp"].dt.hour.astype("int8")
    ordered["day_of_week"] = ordered["timestamp"].dt.dayofweek.astype("int8")
    ordered["time_since_previous_s"] = (
        ordered.groupby("customer_id")["timestamp"].diff().dt.total_seconds().fillna(-1)
    )
    indexed = ordered.set_index("timestamp")
    output = ordered.set_index("event_id")[["hour_of_day", "day_of_week", "time_since_previous_s"]]
    for entity in ("customer_id", "counterparty_id"):
        short = "customer" if entity == "customer_id" else "counterparty"
        grouped = indexed.groupby(entity, sort=False)
        for label, window in WINDOWS.items():
            roll = grouped.rolling(window, closed="left", min_periods=1)
            values = pd.DataFrame({
                f"{short}_count_{label}": roll["amount"].count(),
                f"{short}_sum_{label}": roll["amount"].sum(),
                f"{short}_mean_{label}": roll["amount"].mean(),
                f"{short}_std_{label}": roll["amount"].std(),
            }).reset_index(level=0, drop=True)
            values["event_id"] = indexed.loc[values.index, "event_id"].to_numpy()
            output = output.join(values.set_index("event_id"))
    hist = indexed.groupby("customer_id", sort=False).rolling("30d", closed="left", min_periods=2)["amount"]
    mean = hist.mean().reset_index(level=0, drop=True)
    std = hist.std().reset_index(level=0, drop=True).replace(0, np.nan)
    z = ((indexed.loc[mean.index, "amount"].to_numpy() - mean.to_numpy()) / std.to_numpy())
    z_frame = pd.DataFrame({"event_id": indexed.loc[mean.index, "event_id"].to_numpy(), "amount_zscore_30d": z})
    return output.join(z_frame.set_index("event_id")).reindex(events["event_id"]).fillna(0.0)

