"""LightGBM detector head for A1 features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class A1Detector:
    """Deterministic binary classifier with a safe constant-prior fallback."""

    name = "a1_lightgbm"

    def __init__(self) -> None:
        self._columns: list[str] | None = None
        self._model: LGBMClassifier | None = None
        self._constant_score: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        frame = self._numeric_frame(X)
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be a one-dimensional array")
        if len(frame) != len(labels):
            raise ValueError("X and y must contain the same number of rows")
        if len(frame) == 0:
            raise ValueError("training data cannot be empty")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("y must contain only binary labels 0 and 1")

        self._columns = list(frame.columns)
        classes = np.unique(labels)
        if len(classes) == 1 or len(frame) < 4:
            self._model = None
            self._constant_score = float(np.mean(labels))
            return

        self._constant_score = None
        self._model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        self._model.fit(frame, labels.astype(np.int8, copy=False))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self._columns is None:
            raise RuntimeError("fit() must be called before score()")
        frame = self._aligned_frame(X)
        if self._constant_score is not None:
            scores = np.full(len(frame), self._constant_score, dtype=np.float64)
        elif self._model is not None:
            scores = self._model.predict_proba(frame)[:, 1].astype(np.float64, copy=False)
        else:  # Defensive invariant; normal code cannot reach this branch.
            raise RuntimeError("detector has no fitted scoring state")
        return np.clip(scores, 0.0, 1.0)

    @staticmethod
    def _numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.columns.duplicated().any():
            raise ValueError("feature column names must be unique")
        numeric = X.select_dtypes(include=[np.number]).copy()
        if numeric.shape[1] == 0:
            raise ValueError("X must contain at least one numeric feature column")
        return numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

    def _aligned_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.columns.duplicated().any():
            raise ValueError("feature column names must be unique")
        aligned = pd.DataFrame(index=X.index)
        for column in self._columns or []:
            if column in X and pd.api.types.is_numeric_dtype(X[column]):
                aligned[column] = pd.to_numeric(X[column], errors="coerce")
            else:
                aligned[column] = 0.0
        return aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
