"""Synthetic-identity bust-out attack package."""

from .detector import A4Detector
from .features import A4Features
from .generator import A4Generator

GENERATOR = A4Generator()
FEATURE_BLOCK = A4Features()
DETECTOR = A4Detector()

__all__ = ["DETECTOR", "FEATURE_BLOCK", "GENERATOR"]
