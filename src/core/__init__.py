"""Core types and utilities."""

from .types import BoundingBox, FeatureRect, OCRWord, AnalysisResult
from .loader import load_first_image, get_image_files

__all__ = [
    "BoundingBox",
    "FeatureRect",
    "OCRWord",
    "AnalysisResult",
    "load_first_image",
    "get_image_files",
]
