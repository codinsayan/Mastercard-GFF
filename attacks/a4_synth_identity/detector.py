"""Compact tabular detector for A4 snapshot features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class A4Detector:
    name = "a4_hist_gradient_boosting"

    def __init__(self) -> None:
        self.model = HistGradientBoostingClassifier(
            max_depth=4, max_iter=120, learning_rate=0.08, random_state=42
        )
        self.columns: list[str] | None = None
        self.constant_score: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        if len(X) != len(y) or not len(X):
            raise ValueError("X and y must have equal non-zero length")
        self.columns = list(X.columns)
        labels = np.asarray(y)
        classes = np.unique(labels)
        if len(classes) == 1:
            self.constant_score = float(classes[0])
            return
        self.constant_score = None
        self.model.fit(self._matrix(X), labels)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.columns is None:
            raise RuntimeError("fit() must be called before score()")
        if self.constant_score is not None:
            return np.full(len(X), self.constant_score, dtype=float)
        return self.model.predict_proba(self._matrix(X))[:, 1].astype(float)

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        missing = set(self.columns or []).difference(X.columns)
        if missing:
            raise ValueError(f"detector features are missing: {sorted(missing)}")
        return X.loc[:, self.columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
