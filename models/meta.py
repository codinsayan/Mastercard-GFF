"""Logistic detector-head stacking with an IsolationForest novelty score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MetaLearner:
    """Fuse detector scores and unsupervised novelty into a calibrated risk score."""

    def __init__(self, seed: int = 42) -> None:
        self.novelty = IsolationForest(n_estimators=100, random_state=seed)
        self.scaler = StandardScaler()
        self.stack = LogisticRegression(max_iter=1_000, random_state=seed)
        self.head_columns: list[str] | None = None

    def fit(
        self, head_scores: pd.DataFrame, feature_matrix: pd.DataFrame, labels: np.ndarray
    ) -> None:
        if not (len(head_scores) == len(feature_matrix) == len(labels)) or not len(labels):
            raise ValueError("meta-learner inputs must have equal non-zero lengths")
        self.head_columns = list(head_scores.columns)
        self.novelty.fit(self._finite(feature_matrix))
        novelty_score = -self.novelty.score_samples(self._finite(feature_matrix))
        stacked = np.column_stack((self._finite(head_scores), novelty_score))
        self.stack.fit(self.scaler.fit_transform(stacked), np.asarray(labels, dtype=np.int8))

    def score(self, head_scores: pd.DataFrame, feature_matrix: pd.DataFrame) -> np.ndarray:
        if self.head_columns is None:
            raise RuntimeError("fit() must be called before score()")
        missing = set(self.head_columns).difference(head_scores.columns)
        if missing:
            raise ValueError(f"detector head scores are missing: {sorted(missing)}")
        novelty_score = -self.novelty.score_samples(self._finite(feature_matrix))
        stacked = np.column_stack(
            (self._finite(head_scores.loc[:, self.head_columns]), novelty_score)
        )
        return self.stack.predict_proba(self.scaler.transform(stacked))[:, 1].astype(float)

    @staticmethod
    def _finite(frame: pd.DataFrame) -> np.ndarray:
        return frame.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
