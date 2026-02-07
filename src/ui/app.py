"""Streamlit application for image feature extraction and analysis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.config import TRAIN_DIR, TEMP_DIR, DEFAULT_SENSITIVITY
from src.core import load_first_image, get_image_files, AnalysisResult
from src.core.types import BoundingBox
from src.extraction import detect_feature_rectangles, extract_all_features
from src.analysis import (
    detect_text,
    draw_ocr_overlay,
    calculate_text_coverage,
    compute_12bin_histogram,
    render_histogram_image,
)

MAX_FEATURE_SIZE = 200


def resize_to_max(image: np.ndarray, max_size: int = MAX_FEATURE_SIZE) -> np.ndarray:
    """Resize image to fit within max_size while keeping aspect ratio."""
    h, w = image.shape[:2]
    if h <= max_size and w <= max_size:
        return image

    scale = min(max_size / w, max_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_numbered_boxes(
    image: np.ndarray,
    boxes: list[BoundingBox],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw numbered rectangles on an image.

    Args:
        image: RGB image as numpy array.
        boxes: List of bounding boxes.
        color: RGB color for rectangles.
        thickness: Line thickness.

    Returns:
        Image with numbered rectangles drawn.
    """
    overlay = image.copy()

    for i, box in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(
            overlay,
            (box.x, box.y),
            (box.x + box.w, box.y + box.h),
            color,
            thickness,
        )

        # Draw number label
        label = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        label_thickness = 3

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, label_thickness
        )

        # Draw background rectangle for label
        label_x = box.x
        label_y = max(box.y - 10, text_h + 10)
        cv2.rectangle(
            overlay,
            (label_x - 2, label_y - text_h - 5),
            (label_x + text_w + 5, label_y + 5),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            overlay,
            label,
            (label_x, label_y),
            font,
            font_scale,
            (0, 0, 0),
            label_thickness,
            cv2.LINE_AA,
        )

    return overlay


def run_analysis_pipeline(
    image_path: Path,
    sensitivity: int = DEFAULT_SENSITIVITY,
) -> tuple[np.ndarray, list[AnalysisResult]]:
    """Run the complete analysis pipeline on an image.

    Args:
        image_path: Path to the image to analyze.
        sensitivity: Detection sensitivity (1-10).

    Returns:
        Tuple of (annotated full image, list of AnalysisResult objects).
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return np.array([]), []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect feature rectangles
    boxes = detect_feature_rectangles(image_rgb, sensitivity=sensitivity)

    # Draw numbered boxes on full image
    annotated_image = draw_numbered_boxes(image_rgb, boxes)

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

    return annotated_image, results


def render_table(results: list[AnalysisResult]) -> None:
    """Render analysis results as a Streamlit table.

    Args:
        results: List of AnalysisResult objects to display.
    """
    if not results:
        st.warning("No features detected in the image.")
        return

    # Header
    cols = st.columns([0.5, 2, 1, 1, 2])
    cols[0].markdown("**#**")
    cols[1].markdown("**OCR Overlay**")
    cols[2].markdown("**Size**")
    cols[3].markdown("**Coverage**")
    cols[4].markdown("**Histogram**")

    st.divider()

    # Rows
    for i, result in enumerate(results):
        cols = st.columns([0.5, 2, 1, 1, 2])

        # Feature number
        cols[0].markdown(f"### {i + 1}")

        # OCR overlay (resized)
        resized_overlay = resize_to_max(result.ocr_overlay_image)
        cols[1].image(resized_overlay)

        # Size in pixels
        h, w = result.feature.image.shape[:2]
        cols[2].markdown(f"**{w} x {h}** px")

        # Coverage percentage
        cols[3].metric(
            label="Text Coverage",
            value=f"{result.text_coverage_percent:.1f}%",
        )

        # Histogram
        cols[4].image(result.histogram_image)

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

    # Get available images
    image_files = get_image_files(TRAIN_DIR)

    if not image_files:
        st.error(f"No images found in {TRAIN_DIR}")
        st.info("Please add images to the train directory.")
        return

    # Initialize session state
    if "sensitivity" not in st.session_state:
        st.session_state.sensitivity = DEFAULT_SENSITIVITY

    # Controls row
    col1, col2 = st.columns([2, 2])

    with col1:
        # File dropdown
        file_names = [f.name for f in image_files]
        selected_file = st.selectbox(
            "Select image",
            file_names,
            key="selected_file",
        )
        image_path = TRAIN_DIR / selected_file

    with col2:
        sensitivity = st.slider(
            "Detection Sensitivity",
            min_value=1,
            max_value=10,
            value=st.session_state.sensitivity,
            key="sensitivity",
            help="Higher = more sensitive (finds more features)",
        )

    # Run analysis
    with st.spinner("Extracting and analyzing features..."):
        annotated_image, results = run_analysis_pipeline(image_path, sensitivity)

    st.success(f"Found {len(results)} features")

    # Show annotated image with numbered boxes
    st.image(annotated_image, use_column_width=True)

    st.divider()

    # Render results table
    render_table(results)
