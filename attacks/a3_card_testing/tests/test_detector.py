from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attacks.a3_card_testing.detector import A3Detector


def _training_data() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(
        {
            "a3_signal": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "a3_other": [1.0, np.nan, 0.8, 0.2, np.inf, 0.0],
        }
    )
    return X, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)


def test_detector_fits_deterministically_and_scores_probabilities():
    X, y = _training_data()
    first = A3Detector()
    second = A3Detector()
    first.fit(X, y)
    second.fit(X, y)

    first_scores = first.score(X)
    second_scores = second.score(X)
    np.testing.assert_array_equal(first_scores, second_scores)
    assert first_scores.dtype == np.dtype(float)
    assert first_scores.shape == (len(X),)
    assert np.isfinite(first_scores).all()
    assert ((0.0 <= first_scores) & (first_scores <= 1.0)).all()


def test_detector_rejects_unfitted_or_misaligned_features():
    X, y = _training_data()
    with pytest.raises(RuntimeError, match="fitted"):
        A3Detector().score(X)

    detector = A3Detector()
    detector.fit(X, y)
    with pytest.raises(ValueError, match="columns and order"):
        detector.score(X.loc[:, list(reversed(X.columns))])


def test_detector_supports_single_class_training_slices():
    X, _ = _training_data()
    detector = A3Detector()
    detector.fit(X, np.ones(len(X), dtype=np.int8))

    np.testing.assert_array_equal(detector.score(X), np.ones(len(X), dtype=float))
