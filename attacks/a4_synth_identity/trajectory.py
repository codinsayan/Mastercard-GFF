"""Unsupervised weekly GRU embeddings and behavioral trajectory clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from torch import nn

WEEKS = 13
INPUT_SIZE = 5
HIDDEN_SIZE = 32


class _TrajectoryAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(HIDDEN_SIZE, INPUT_SIZE)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, hidden = self.encoder(values)
        return self.decoder(encoded), hidden[-1]


def weekly_sequences(events: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    """Summarize each account's first 90 days into 13 weekly five-value steps."""
    required = {
        "customer_id",
        "counterparty_id",
        "amount",
        "day",
        "utilization",
        "payment",
    }
    if missing := required.difference(events.columns):
        raise ValueError(f"trajectory inputs are missing: {sorted(missing)}")
    work = events.loc[events["day"].between(0, 89)].copy()
    work["week"] = (work["day"] // 7).clip(0, WEEKS - 1).astype(int)
    customers = pd.Index(events["customer_id"].drop_duplicates(), name="customer_id")
    sequences = np.zeros((len(customers), WEEKS, INPUT_SIZE), dtype=np.float32)
    positions = {customer: index for index, customer in enumerate(customers)}

    weekly = work.groupby(["customer_id", "week"], sort=False).agg(
        transaction_count=("amount", "size"),
        mean_amount=("amount", "mean"),
        utilization=("utilization", "last"),
        payment_ratio=("payment", "mean"),
        distinct_merchants=("counterparty_id", "nunique"),
    )
    for (customer, week), row in weekly.iterrows():
        sequences[positions[customer], int(week)] = row.to_numpy(dtype=np.float32)

    for feature in range(INPUT_SIZE):
        values = sequences[:, :, feature]
        nonzero = values != 0
        if nonzero.any():
            mean = float(values[nonzero].mean())
            std = max(float(values[nonzero].std()), 1e-6)
            values[nonzero] = (values[nonzero] - mean) / std
    return customers, sequences


def trajectory_features(events: pd.DataFrame, *, seed: int = 42, epochs: int = 4) -> pd.DataFrame:
    """Train a label-free GRU autoencoder and cluster its account embeddings."""
    if epochs < 0:
        raise ValueError("epochs cannot be negative")
    customers, sequences = weekly_sequences(events)
    if not len(customers):
        return pd.DataFrame(index=customers)

    if epochs:
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        values = torch.from_numpy(sequences)
        model = _TrajectoryAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = model(values)
            loss = nn.functional.mse_loss(reconstructed, values)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, embedding = model(values)
        matrix = embedding.numpy()
    else:
        # Fast CEM path: the documented weekly-statistics fallback, never used for final metrics.
        matrix = sequences.reshape(len(customers), -1)
    scale = matrix.std(axis=0)
    standardized = (matrix - matrix.mean(axis=0)) / np.where(scale > 1e-6, scale, 1)

    if len(customers) < 4:
        labels = np.zeros(len(customers), dtype=int)
    else:
        min_samples = max(3, min(8, int(np.sqrt(len(customers)) / 2)))
        labels = DBSCAN(eps=1.6, min_samples=min_samples).fit_predict(standardized)

    sizes = pd.Series(labels[labels >= 0]).value_counts().to_dict()
    cluster_size = np.array([sizes.get(label, 1) for label in labels], dtype=float)
    centroid_distance = np.zeros(len(customers), dtype=float)
    for label in set(labels):
        positions = np.flatnonzero(labels == label)
        if label < 0:
            continue
        centroid = standardized[positions].mean(axis=0)
        centroid_distance[positions] = np.sqrt(
            ((standardized[positions] - centroid) ** 2).mean(axis=1)
        )
    return pd.DataFrame(
        {
            "cluster_density": (cluster_size - 1) / max(1, len(customers) - 1),
            "centroid_distance": centroid_distance,
            "cluster_size": cluster_size,
            "cluster_label": labels,
        },
        index=customers,
    )
