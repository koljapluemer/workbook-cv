"""Data types for image feature extraction and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BoundingBox:
    """Represents a bounding box with position and dimensions."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        """Calculate the area of the bounding box."""
        return self.w * self.h


@dataclass
class FeatureRect:
    """Represents an extracted feature rectangle."""

    box: BoundingBox
    image: np.ndarray
    temp_path: Path


@dataclass
class OCRWord:
    """Represents a detected word from OCR."""

    text: str
    bbox: BoundingBox
    confidence: float


@dataclass
class AnalysisResult:
    """Contains all analysis results for a single feature."""

    feature: FeatureRect
    ocr_overlay_image: np.ndarray
    ocr_words: list[OCRWord] = field(default_factory=list)
    text_coverage_percent: float = 0.0
    histogram_image: np.ndarray = field(default_factory=lambda: np.array([]))
