"""Streamlit application for image feature extraction and analysis."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import TRAIN_DIR, TEMP_DIR, DEFAULT_SENSITIVITY
from src.core import load_first_image, AnalysisResult
from src.extraction import detect_feature_rectangles, extract_all_features
from src.analysis import (
    detect_text,
    draw_ocr_overlay,
    calculate_text_coverage,
    compute_12bin_histogram,
    render_histogram_image,
)


def run_analysis_pipeline(image_path: Path) -> list[AnalysisResult]:
    """Run the complete analysis pipeline on an image.

    Args:
        image_path: Path to the image to analyze.

    Returns:
        List of AnalysisResult objects for each detected feature.
    """
    import cv2

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect feature rectangles
    boxes = detect_feature_rectangles(image_rgb, sensitivity=DEFAULT_SENSITIVITY)

    # Extract features
    features = extract_all_features(image_rgb, boxes, TEMP_DIR)

    # Analyze each feature
    results = []
    for feature in features:
        # Run OCR
        ocr_words = detect_text(feature.image)

        # Draw overlay
        ocr_overlay = draw_ocr_overlay(feature.image, ocr_words)

        # Calculate coverage
        coverage = calculate_text_coverage(feature.image.shape, ocr_words)

        # Compute histogram
        histogram = compute_12bin_histogram(feature.image)
        histogram_img = render_histogram_image(histogram)

        result = AnalysisResult(
            feature=feature,
            ocr_overlay_image=ocr_overlay,
            ocr_words=ocr_words,
            text_coverage_percent=coverage,
            histogram_image=histogram_img,
        )
        results.append(result)

    return results


def render_table(results: list[AnalysisResult]) -> None:
    """Render analysis results as a Streamlit table.

    Args:
        results: List of AnalysisResult objects to display.
    """
    if not results:
        st.warning("No features detected in the image.")
        return

    # Header
    cols = st.columns([1, 1, 1, 1])
    cols[0].markdown("**Original**")
    cols[1].markdown("**OCR Overlay**")
    cols[2].markdown("**Coverage**")
    cols[3].markdown("**Histogram**")

    st.divider()

    # Rows
    for i, result in enumerate(results):
        cols = st.columns([1, 1, 1, 1])

        # Original feature
        cols[0].image(result.feature.image, use_column_width=True)

        # OCR overlay
        cols[1].image(result.ocr_overlay_image, use_column_width=True)

        # Coverage percentage
        cols[2].metric(
            label="Text Coverage",
            value=f"{result.text_coverage_percent:.1f}%",
        )

        # Histogram
        cols[3].image(result.histogram_image, use_column_width=True)

        if i < len(results) - 1:
            st.divider()


def run_app() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Image Feature Extraction",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Image Feature Extraction & Analysis")

    # Load first image from train directory
    result = load_first_image(TRAIN_DIR)

    if result is None:
        st.error(f"No images found in {TRAIN_DIR}")
        st.info("Please add images to the train directory.")
        return

    image, image_path = result

    st.subheader(f"Analyzing: {image_path.name}")

    # Show original image
    with st.expander("Original Image", expanded=False):
        st.image(image, use_column_width=True)

    # Run analysis
    with st.spinner("Extracting and analyzing features..."):
        results = run_analysis_pipeline(image_path)

    st.success(f"Found {len(results)} features")

    # Render results table
    render_table(results)
