"""Frozen interfaces plus explicit attack-component registration."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class AttackGenerator(Protocol):
    attack_id: str
    def default_params(self) -> dict: ...
    def param_space(self) -> dict[str, tuple[float, float]]: ...
    def generate(self, params: dict, population: Any, seed: int) -> pd.DataFrame: ...
    def utility(self, df: pd.DataFrame) -> float: ...


@runtime_checkable
class FeatureBlock(Protocol):
    attack_id: str
    def build(self, events: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class DetectorHead(Protocol):
    name: str
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None: ...
    def score(self, X: pd.DataFrame) -> np.ndarray: ...


_ATTACKS: dict[str, tuple[AttackGenerator, FeatureBlock, DetectorHead]] = {}


def register(generator: AttackGenerator, features: FeatureBlock, detector: DetectorHead) -> None:
    attack_id = generator.attack_id.upper()
    if attack_id not in {"A1", "A2", "A3", "A4"}:
        raise ValueError(f"Unsupported attack id: {attack_id}")
    if features.attack_id.upper() != attack_id:
        raise ValueError("generator and feature block attack ids differ")
    if attack_id in _ATTACKS:
        raise ValueError(f"Attack already registered: {attack_id}")
    _ATTACKS[attack_id] = (generator, features, detector)


def registered_attacks() -> dict[str, tuple[AttackGenerator, FeatureBlock, DetectorHead]]:
    return dict(_ATTACKS)

