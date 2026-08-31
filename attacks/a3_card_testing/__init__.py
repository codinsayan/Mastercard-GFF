"""Public A3 attack components used by the shared registry and round runner."""

from .detector import A3Detector
from .features import A3Features
from .generator import A3Generator

GENERATOR = A3Generator()
FEATURE_BLOCK = A3Features()
DETECTOR = A3Detector()

__all__ = ["DETECTOR", "FEATURE_BLOCK", "GENERATOR"]
