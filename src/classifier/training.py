"""Training module for the feature classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .types import ClassifierResult


def extract_features_from_label(entry: dict) -> np.ndarray:
    """Extract the 15 features from a label entry.

    Features: coverage, width, height, 12 histogram bins.
    """
    features = entry["features"]
    return np.array([
        features["text_coverage_percent"],
        features["width"],
        features["height"],
        *features["histogram"],
    ])


def train_classifier(labels_path: Path) -> ClassifierResult | None:
    """Train a RandomForest classifier on labeled data.

    Args:
        labels_path: Path to the labels JSON file.

    Returns:
        ClassifierResult with trained model, or None if no labeled data.
    """
    if not labels_path.exists():
        return None

    with open(labels_path) as f:
        data = json.load(f)

    label_entries = data.get("labels", [])
    if not label_entries:
        return None

    X = []
    y = []
    for entry in label_entries:
        X.append(extract_features_from_label(entry))
        y.append(entry["label"])

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    unique_labels = sorted(set(y))

    return ClassifierResult(
        model=model,
        n_samples=len(y),
        labels=unique_labels,
    )
