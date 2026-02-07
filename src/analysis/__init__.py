"""Analysis modules for OCR, coverage, and histograms."""

from .ocr import detect_text, draw_ocr_overlay
from .coverage import calculate_text_coverage
from .histogram import compute_12bin_histogram, render_histogram_image

__all__ = [
    "detect_text",
    "draw_ocr_overlay",
    "calculate_text_coverage",
    "compute_12bin_histogram",
    "render_histogram_image",
]
