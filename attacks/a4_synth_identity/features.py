"""Snapshot-level static and behavioral features for synthetic-identity accounts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .trajectory import trajectory_features

FEATURE_NAMES = (
    "a4_shared_address_token_count",
    "a4_shared_device_prefix_count",
    "a4_phone_block_density",
    "a4_account_age_at_first_large_txn",
    "a4_trajectory_cluster_density",
    "a4_trajectory_dist_to_centroid",
    "a4_trajectory_cluster_size",
    "a4_ramp_convexity",
    "a4_payment_ratio_regularity",
    "a4_bustout_sync_score",
)


class A4Features:
    attack_id = "A4"

    def __init__(self, *, trajectory_epochs: int = 4) -> None:
        self.trajectory_epochs = trajectory_epochs

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        """Build campaign-snapshot features and repeat account values for its events."""
        required = {"event_id", "customer_id", "counterparty_id", "amount", "meta"}
        if missing := required.difference(events.columns):
            raise ValueError(f"A4 feature inputs are missing: {sorted(missing)}")
        if events["event_id"].duplicated().any():
            raise ValueError("event_id must be unique")
        if events.empty:
            return pd.DataFrame(columns=FEATURE_NAMES, index=pd.Index([], name="event_id"))

        work = events.copy()
        keys = {
            "address": "a4_address_token",
            "device": "a4_device_prefix",
            "phone": "a4_phone_block",
            "day": "a4_days_since_onboarding",
            "limit": "a4_credit_limit",
            "utilization": "a4_utilization",
            "payment": "a4_payment_ratio",
        }
        for column, key in keys.items():
            work[column] = work["meta"].map(lambda meta, item=key: meta.get(item, 0))
        numeric = ("day", "limit", "utilization", "payment")
        work[list(numeric)] = work[list(numeric)].apply(pd.to_numeric, errors="coerce").fillna(0)

        accounts = work.sort_values("day").groupby("customer_id", sort=False).first()
        account_features = pd.DataFrame(index=accounts.index)
        account_count = len(accounts)

        def shared_count(column: str) -> pd.Series:
            values = accounts[column].astype(str)
            counts = values[values.ne("0") & values.ne("")].value_counts()
            return values.map(counts).fillna(1).sub(1).clip(lower=0).astype(float)

        account_features[FEATURE_NAMES[0]] = shared_count("address")
        account_features[FEATURE_NAMES[1]] = shared_count("device")
        phone_count = shared_count("phone")
        account_features[FEATURE_NAMES[2]] = phone_count / max(1, account_count - 1)

        summaries = []
        for customer_id, group in work.groupby("customer_id", sort=False):
            group = group.sort_values(["day", "event_id"])
            day = group["day"].to_numpy(dtype=float)
            utilization = group["utilization"].to_numpy(dtype=float)
            payment = group["payment"].to_numpy(dtype=float)
            limit = group["limit"].replace(0, np.nan).to_numpy(dtype=float)
            # The build window is everything before this account's first utilization spike,
            # not every row that happens to sit under the threshold. A draw-down opens below
            # 0.85 before crossing it, so a fixed threshold readmits the first draw-down row
            # -- and its zero repayment -- into the account's own build statistics, which
            # made payment regularity a detector of missed payments instead. Derived from
            # observed utilization only; the generator's phase label is never read.
            spike = day[utilization >= 0.85]
            spike_day = float(spike.min()) if len(spike) else np.nan
            build = (
                day < spike_day
                if np.isfinite(spike_day)
                else np.ones(len(day), dtype=bool)
            )
            build_day = day[build]
            build_utilization = utilization[build]
            if len(np.unique(build_day)) >= 3:
                x = (build_day - build_day.min()) / (build_day.max() - build_day.min())
                convexity = float(np.polyfit(x, build_utilization, 2)[0])
            else:
                convexity = 0.0
            build_payment = payment[build]
            payment_std = float(np.std(build_payment)) if len(build_payment) else 0.0
            regularity = 1 / (1 + 100 * payment_std)
            large = group["amount"].to_numpy(dtype=float) >= 0.2 * np.nan_to_num(limit)
            first_large_day = float(day[large].min()) if large.any() else float(day.max())
            summaries.append((customer_id, first_large_day, convexity, regularity, spike_day))

        summary = pd.DataFrame(
            summaries,
            columns=["customer_id", "first_large", "convexity", "regularity", "spike_day"],
        ).set_index("customer_id")
        trajectory = trajectory_features(work, epochs=self.trajectory_epochs).reindex(summary.index)
        spike_days = summary["spike_day"].to_numpy(dtype=float)
        synchronized = np.zeros(account_count)
        labels = trajectory["cluster_label"].to_numpy(dtype=int)
        for index, label in enumerate(labels):
            same_cluster = labels == label if label >= 0 else np.arange(account_count) == index
            comparable = same_cluster & np.isfinite(spike_days)
            close = comparable & (np.abs(spike_days - spike_days[index]) <= 7)
            synchronized[index] = max(0, close.sum() - 1) / max(1, comparable.sum() - 1)

        account_features = account_features.reindex(summary.index)
        account_features[FEATURE_NAMES[3]] = summary["first_large"]
        account_features[FEATURE_NAMES[4]] = trajectory["cluster_density"]
        account_features[FEATURE_NAMES[5]] = trajectory["centroid_distance"]
        account_features[FEATURE_NAMES[6]] = trajectory["cluster_size"]
        account_features[FEATURE_NAMES[7]] = summary["convexity"]
        account_features[FEATURE_NAMES[8]] = summary["regularity"]
        account_features[FEATURE_NAMES[9]] = synchronized

        output = account_features.reindex(work["customer_id"]).set_axis(
            work["event_id"].astype("string")
        )
        output.index.name = "event_id"
        return output.loc[:, FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
