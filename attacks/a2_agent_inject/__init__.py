"""A2 prompt-injected shopping-agent attack package."""

from .detector import Detector
from .features import Features
from .generator import Generator

GENERATOR = Generator()
FEATURE_BLOCK = Features()
DETECTOR = Detector()

__all__ = ["DETECTOR", "FEATURE_BLOCK", "GENERATOR", "Detector", "Features", "Generator"]

