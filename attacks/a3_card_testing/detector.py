"""LightGBM detector head for A3 card-testing features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class A3Detector:
    """Deterministic binary classifier implementing the shared DetectorHead contract."""

    name = "a3_lightgbm"

    def __init__(self) -> None:
        self._model: LGBMClassifier | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._constant_score: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        values = self._validated_features(X, fitting=True)
        targets = np.asarray(y)
        if targets.ndim != 1 or len(targets) != len(values):
            raise ValueError("y must be a one-dimensional array aligned with X")
        if not np.isin(targets, (0, 1)).all():
            raise ValueError("y must contain only binary labels 0 and 1")
        if not len(targets):
            raise ValueError("X and y must not be empty")

        self._feature_names = tuple(X.columns)
        unique = np.unique(targets)
        if len(unique) == 1:
            self._model = None
            self._constant_score = float(unique[0])
            return

        self._constant_score = None
        self._model = LGBMClassifier(
            objective="binary",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        self._model.fit(values, targets.astype(np.int8, copy=False))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self._feature_names is None:
            raise RuntimeError("detector must be fitted before scoring")
        if tuple(X.columns) != self._feature_names:
            raise ValueError("scoring feature columns and order must match fitted features")
        values = self._validated_features(X, fitting=False)
        if self._constant_score is not None:
            return np.full(len(values), self._constant_score, dtype=float)
        if self._model is None:  # pragma: no cover - protected by fitted state above
            raise RuntimeError("detector has no fitted model")
        scores = self._model.predict_proba(values)[:, 1].astype(float, copy=False)
        return np.clip(scores, 0.0, 1.0)

    @staticmethod
    def _validated_features(X: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if fitting and X.columns.duplicated().any():
            raise ValueError("X feature names must be unique")
        if fitting and not len(X.columns):
            raise ValueError("X must contain at least one feature")
        if not all(str(column).startswith("a3_") for column in X.columns):
            raise ValueError("all detector features must use the a3_ prefix")
        numeric = X.apply(pd.to_numeric, errors="coerce").astype(float)
        return numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
