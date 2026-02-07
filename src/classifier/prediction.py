"""Prediction module for classifying features."""

from __future__ import annotations

import numpy as np

from src.analysis import compute_12bin_histogram
from src.core import AnalysisResult

from .types import ClassifierResult, PredictionResult


def extract_features_from_result(result: AnalysisResult) -> np.ndarray:
    """Extract the 15 features from an AnalysisResult.

    Features: coverage, width, height, 12 histogram bins.
    """
    h, w = result.feature.image.shape[:2]
    histogram = compute_12bin_histogram(result.feature.image)
    return np.array([
        result.text_coverage_percent,
        w,
        h,
        *histogram,
    ])


def predict_features(
    results: list[AnalysisResult],
    classifier: ClassifierResult,
) -> list[PredictionResult]:
    """Predict labels for analysis results.

    Args:
        results: List of AnalysisResult objects to classify.
        classifier: Trained ClassifierResult.

    Returns:
        List of PredictionResult with predicted labels and confidence.
    """
    if not results:
        return []

    X = np.array([extract_features_from_result(r) for r in results])
    predictions = classifier.model.predict(X)
    probabilities = classifier.model.predict_proba(X)

    prediction_results = []
    for pred, probs in zip(predictions, probabilities):
        confidence = max(probs)
        prediction_results.append(PredictionResult(
            label=pred,
            confidence=confidence,
        ))

    return prediction_results
