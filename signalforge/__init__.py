"""Public, sanitized building blocks for SignalForge.

This package intentionally excludes production feature definitions,
trained weights, private selectors, vendor credentials, and live outputs.
"""

from .regime_film import RegimeFiLM
from .training import LossCoefficients, TrainingBatch, evaluate_one_epoch, train_one_epoch

__all__ = [
    "RegimeFiLM",
    "LossCoefficients",
    "TrainingBatch",
    "evaluate_one_epoch",
    "train_one_epoch",
]
