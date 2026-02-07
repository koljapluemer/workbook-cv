"""Data types for feature labeling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureData:
    """Stores extracted feature measurements for ML training."""

    text_coverage_percent: float
    width: int
    height: int
    histogram: list[int]  # 12 values (R0-R3, G0-G3, B0-B3)


@dataclass
class LabeledFeature:
    """A labeled feature with its classification and measurements."""

    id: str  # "{image_name}_{feature_index}"
    image_file: str
    feature_index: int
    label: str  # "label" | "figure" | "irrelevant"
    features: FeatureData
