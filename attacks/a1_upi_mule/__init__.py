"""A1 deepfake-coercion UPI mule attack integration exports."""

from .detector import A1Detector
from .features import A1Features
from .generator import A1Generator

GENERATOR = A1Generator()
FEATURE_BLOCK = A1Features()
DETECTOR = A1Detector()

__all__ = [
    "DETECTOR",
    "FEATURE_BLOCK",
    "GENERATOR",
    "A1Detector",
    "A1Features",
    "A1Generator",
]
