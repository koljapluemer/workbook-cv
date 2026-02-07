"""Type definitions for the classifier module."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier


@dataclass
class ClassifierResult:
    """Result of training a classifier."""

    model: RandomForestClassifier
    n_samples: int
    labels: list[str]


@dataclass
class PredictionResult:
    """Prediction result for a single feature."""

    label: str
    confidence: float
