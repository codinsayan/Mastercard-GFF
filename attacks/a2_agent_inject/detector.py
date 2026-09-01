"""LightGBM detector head for A2 agentic-commerce features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class Detector:
    name = "a2_semantic_economic_lightgbm"

    def __init__(
        self,
        feature_names: Sequence[str] | None = None,
        *,
        random_state: int = 42,
    ) -> None:
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self.random_state = random_state
        self._columns: tuple[str, ...] | None = None
        self._model = None
        self._constant_score: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        from lightgbm import LGBMClassifier

        prepared = self._prepare_for_fit(X)
        labels = np.asarray(y, dtype=np.int8)
        if len(prepared) != len(labels):
            raise ValueError("X and y must contain the same number of rows")
        classes = np.unique(labels)
        if not set(classes).issubset({0, 1}):
            raise ValueError("detector labels must be binary")
        if len(classes) == 1:
            self._model = None
            self._constant_score = float(classes[0])
            return
        self._constant_score = None
        self._model = LGBMClassifier(
            objective="binary",
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=12,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.5,
            random_state=self.random_state,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        self._model.fit(prepared, labels)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self._columns is None:
            raise RuntimeError("fit() must be called before score()")
        prepared = X.reindex(columns=self._columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self._constant_score is not None:
            return np.full(len(prepared), self._constant_score, dtype=float)
        if self._model is None:
            raise RuntimeError("detector has no fitted model")
        probabilities = np.asarray(self._model.predict_proba(prepared)[:, 1], dtype=float)
        return np.clip(probabilities, 0.0, 1.0)

    def _prepare_for_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or X.columns.empty:
            raise ValueError("X must be a pandas DataFrame with at least one feature")
        columns = self.feature_names or tuple(X.columns)
        missing = sorted(set(columns).difference(X.columns))
        if missing:
            raise ValueError(f"detector features are missing: {missing}")
        self._columns = tuple(columns)
        return X.loc[:, self._columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
