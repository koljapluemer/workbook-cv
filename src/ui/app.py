"""Streamlit application for image feature extraction and analysis."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import datetime

import re

from src.config import (
    TRAIN_DIR, TEMP_DIR, DEFAULT_SENSITIVITY, LABELS_FILE, VALIDATE_DIR,
    SETTINGS_FILE, VALIDATION_STORE_DIR,
    HEATMAP_OUTPUT_DIR, HEATMAP_RGB_DIR, HEATMAP_LABELS_DIR,
    HEATMAP_DEFAULT_KERNEL, HEATMAP_DEFAULT_HUE, HEATMAP_DEFAULT_SAT, HEATMAP_DEFAULT_VAL,
)
from src.heatmap import process_image as _generate_heatmap
from src.core import load_first_image, get_image_files, AnalysisResult
from src.core.types import BoundingBox, FeatureRect
from src.extraction import detect_feature_rectangles, extract_all_features
from src.analysis import (
    detect_text,
    draw_ocr_overlay,
    calculate_text_coverage,
    compute_12bin_histogram,
    render_histogram_image,
)
from src.labeling import load_labels, save_labels, add_label, get_labels_for_image
from src.classifier import train_classifier, predict_features, PredictionResult

MAX_FEATURE_SIZE = 200

DEFAULT_SETTINGS = {
    "sensitivity": DEFAULT_SENSITIVITY,
    "merge_horizontal": 0,
    "merge_vertical": 0,
    "heatmap_kernel": HEATMAP_DEFAULT_KERNEL,
    "heatmap_skip_ocr": False,
    "heatmap_hue": HEATMAP_DEFAULT_HUE,
    "heatmap_sat": HEATMAP_DEFAULT_SAT,
    "heatmap_val": HEATMAP_DEFAULT_VAL,
}


def _on_setting_change() -> None:
    """Persist all UI settings from session_state to disk (shared on_change callback)."""
    save_settings({k: st.session_state.get(k, v) for k, v in DEFAULT_SETTINGS.items()})


def load_settings() -> dict:
    """Load UI settings from disk."""
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """Save UI settings to disk."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


# BGR colors for annotated full images
_BOX_COLORS = {
    "label": (255, 100, 0),    # blue-ish
    "figure": (0, 165, 255),   # orange
}
CONFIDENCE_THRESHOLD = 0.70


def _draw_prediction_boxes(
    image_bgr: np.ndarray,
    results: list[AnalysisResult],
    predictions: list,
) -> np.ndarray:
    """Draw colored boxes for confident label/figure predictions on a full image."""
    annotated = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for result, prediction in zip(results, predictions):
        if prediction.label not in _BOX_COLORS:
            continue
        if prediction.confidence < CONFIDENCE_THRESHOLD:
            continue

        color = _BOX_COLORS[prediction.label]
        box = result.feature.box
        cv2.rectangle(annotated, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 3)

        text = f"{prediction.label} {prediction.confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        label_y = max(box.y - 6, th + 6)
        cv2.rectangle(annotated, (box.x, label_y - th - 4), (box.x + tw + 4, label_y + 2), color, -1)
        cv2.putText(annotated, text, (box.x + 2, label_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def save_validation_results(
    results_by_image: list[tuple[Path, list[AnalysisResult]]],
    predictions_by_image: list[list],
    n_labeled_samples: int,
) -> None:
    """Persist validation results (images + metadata) to disk.

    Args:
        results_by_image: List of (source_image_path, analysis_results) per image.
        predictions_by_image: Corresponding list of prediction lists.
        n_labeled_samples: Number of labeled samples used to train.
    """
    img_dir = VALIDATION_STORE_DIR / "images"
    overlay_dir = VALIDATION_STORE_DIR / "overlays"
    hist_dir = VALIDATION_STORE_DIR / "histograms"
    annotated_dir = VALIDATION_STORE_DIR / "annotated"
    for d in (img_dir, overlay_dir, hist_dir, annotated_dir):
        d.mkdir(parents=True, exist_ok=True)

    entries = []
    for (image_path, results), predictions in zip(results_by_image, predictions_by_image):
        stem = image_path.stem

        # Save full annotated image
        full_image = cv2.imread(str(image_path))
        if full_image is not None:
            annotated = _draw_prediction_boxes(full_image, results, predictions)
            cv2.imwrite(str(annotated_dir / f"{stem}.png"), annotated)

        for feature_index, (result, prediction) in enumerate(zip(results, predictions)):
            key = f"{stem}_{feature_index}"
            box = result.feature.box

            # Save feature images
            feat_path = img_dir / f"{key}.png"
            overlay_path = overlay_dir / f"{key}.png"
            hist_path = hist_dir / f"{key}.png"

            cv2.imwrite(str(feat_path), cv2.cvtColor(result.feature.image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(overlay_path), cv2.cvtColor(result.ocr_overlay_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(hist_path), result.histogram_image)

            entries.append({
                "source_image": image_path.name,
                "feature_index": feature_index,
                "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
                "text_coverage_percent": result.text_coverage_percent,
                "predicted_label": prediction.label,
                "confidence": prediction.confidence,
                "feature_image_path": str(feat_path),
                "overlay_image_path": str(overlay_path),
                "histogram_image_path": str(hist_path),
            })

    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_labeled_samples": n_labeled_samples,
        "entries": entries,
    }
    with open(VALIDATION_STORE_DIR / "results.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_validation_results() -> tuple[list[AnalysisResult], list] | None:
    """Load persisted validation results from disk.

    Returns:
        Tuple of (analysis_results, prediction_results), or None if no saved results.
    """
    from src.classifier import PredictionResult

    metadata_path = VALIDATION_STORE_DIR / "results.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    results = []
    predictions = []
    for entry in metadata["entries"]:
        feat_img = cv2.imread(entry["feature_image_path"])
        overlay_img = cv2.imread(entry["overlay_image_path"])
        hist_img = cv2.imread(entry["histogram_image_path"])

        if feat_img is None or overlay_img is None or hist_img is None:
            continue  # skip if images missing from disk

        feat_rgb = cv2.cvtColor(feat_img, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

        b = entry["box"]
        box = BoundingBox(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
        feature = FeatureRect(
            box=box,
            image=feat_rgb,
            temp_path=Path(entry["feature_image_path"]),
        )
        result = AnalysisResult(
            feature=feature,
            ocr_overlay_image=overlay_rgb,
            text_coverage_percent=entry["text_coverage_percent"],
            histogram_image=hist_img,
        )
        results.append(result)
        predictions.append(PredictionResult(
            label=entry["predicted_label"],
            confidence=entry["confidence"],
        ))

    return results, predictions


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
    merge_horizontal: int = 0,
    merge_vertical: int = 0,
) -> tuple[np.ndarray, list[AnalysisResult]]:
    """Run the complete analysis pipeline on an image.

    Args:
        image_path: Path to the image to analyze.
        sensitivity: Detection sensitivity (1-10).
        merge_horizontal: Horizontal gap for merging boxes (pixels).
        merge_vertical: Vertical gap for merging boxes (pixels).

    Returns:
        Tuple of (annotated full image, list of AnalysisResult objects).
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return np.array([]), []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect feature rectangles
    boxes = detect_feature_rectangles(
        image_rgb,
        sensitivity=sensitivity,
        merge_horizontal=merge_horizontal,
        merge_vertical=merge_vertical,
    )

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


def render_table(
    results: list[AnalysisResult],
    image_file: str,
    existing_labels: dict[int, str],
) -> None:
    """Render analysis results as a Streamlit table with labeling.

    Args:
        results: List of AnalysisResult objects to display.
        image_file: Name of the current image file.
        existing_labels: Dict mapping feature index to existing label.
    """
    if not results:
        st.warning("No features detected in the image.")
        return

    # Header
    cols = st.columns([0.5, 2, 1, 1, 2, 1])
    cols[0].markdown("**#**")
    cols[1].markdown("**OCR Overlay**")
    cols[2].markdown("**Size**")
    cols[3].markdown("**Coverage**")
    cols[4].markdown("**Histogram**")
    cols[5].markdown("**Label**")

    st.divider()

    label_options = ["label", "figure", "irrelevant"]
    image_name = Path(image_file).stem

    # Rows
    for i, result in enumerate(results):
        cols = st.columns([0.5, 2, 1, 1, 2, 1])

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

        # Label selection
        current_label = existing_labels.get(i)
        default_idx = label_options.index(current_label) if current_label in label_options else None

        with cols[5]:
            st.radio(
                "Label",
                options=label_options,
                index=default_idx,
                key=f"label_{image_name}_{i}",
                horizontal=True,
                label_visibility="collapsed",
            )

        if i < len(results) - 1:
            st.divider()


def render_prediction_table(
    results: list[AnalysisResult],
    predictions: list[PredictionResult],
) -> None:
    """Render analysis results with predictions (read-only).

    Args:
        results: List of AnalysisResult objects to display.
        predictions: List of PredictionResult with predicted labels.
    """
    if not results:
        st.warning("No features detected in the image.")
        return

    # Header
    cols = st.columns([0.5, 2, 1, 1, 2, 1.5])
    cols[0].markdown("**#**")
    cols[1].markdown("**OCR Overlay**")
    cols[2].markdown("**Size**")
    cols[3].markdown("**Coverage**")
    cols[4].markdown("**Histogram**")
    cols[5].markdown("**Prediction**")

    st.divider()

    # Label emoji mapping
    label_emoji = {
        "label": "L",
        "figure": "F",
        "irrelevant": "X",
    }

    # Rows
    for i, (result, prediction) in enumerate(zip(results, predictions)):
        cols = st.columns([0.5, 2, 1, 1, 2, 1.5])

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

        # Prediction with confidence
        emoji = label_emoji.get(prediction.label, "?")
        cols[5].markdown(
            f"**[{emoji}] {prediction.label}**\n\n"
            f"*{prediction.confidence * 100:.1f}% confidence*"
        )

        if i < len(results) - 1:
            st.divider()


# ---------------------------------------------------------------------------
# Heatmap labeling helpers
# ---------------------------------------------------------------------------

HEATMAP_CANVAS_WIDTH = 900

HEATMAP_CLASSES: dict[int, dict] = {
    0: {"name": "drawing",   "stroke": "#ff4444", "fill": "rgba(255,68,68,0.25)"},
    1: {"name": "textlabel", "stroke": "#4488ff", "fill": "rgba(68,136,255,0.25)"},
}
_CLASS_BY_NAME  = {v["name"]: k for k, v in HEATMAP_CLASSES.items()}
_CLASS_BY_STROKE = {v["stroke"]: k for k, v in HEATMAP_CLASSES.items()}


def load_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO-format label file. Returns list of (class_id, cx, cy, w, h)."""
    if not path.exists():
        return []
    labels = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) == 5:
            labels.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return labels


def save_yolo_labels(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    """Write YOLO-format label file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labels]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def boxes_to_fabric(
    labels: list[tuple[int, float, float, float, float]],
    canvas_w: int,
    canvas_h: int,
    img_w: int,
    img_h: int,
) -> dict:
    """Convert YOLO labels to a Fabric.js initial_drawing dict for st_canvas."""
    sx = canvas_w / img_w
    sy = canvas_h / img_h
    objects = []
    for class_id, cx, cy, nw, nh in labels:
        cls = HEATMAP_CLASSES.get(class_id, HEATMAP_CLASSES[0])
        left   = (cx - nw / 2) * img_w * sx
        top    = (cy - nh / 2) * img_h * sy
        width  = nw * img_w * sx
        height = nh * img_h * sy
        objects.append({
            "type": "rect",
            "version": "4.4.0",
            "originX": "left",
            "originY": "top",
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "fill": cls["fill"],
            "stroke": cls["stroke"],
            "strokeWidth": 2,
            "strokeUniform": True,
            "selectable": True,
        })
    return {"version": "4.4.0", "objects": objects}


def canvas_to_boxes(
    json_data: dict,
    canvas_w: int,
    canvas_h: int,
    img_w: int,
    img_h: int,
    default_class_id: int,
) -> list[tuple[int, float, float, float, float]]:
    """Parse Fabric.js canvas JSON back to YOLO-format labels."""
    sx = img_w / canvas_w
    sy = img_h / canvas_h
    labels = []
    for obj in json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue
        stroke = obj.get("stroke", "")
        class_id = _CLASS_BY_STROKE.get(stroke, default_class_id)

        left   = obj["left"]
        top    = obj["top"]
        width  = obj["width"]  * obj.get("scaleX", 1.0)
        height = obj["height"] * obj.get("scaleY", 1.0)

        img_x  = left   * sx
        img_y  = top    * sy
        img_bw = width  * sx
        img_bh = height * sy

        cx = (img_x + img_bw / 2) / img_w
        cy = (img_y + img_bh / 2) / img_h
        nw = img_bw / img_w
        nh = img_bh / img_h

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        if nw > 0 and nh > 0:
            labels.append((class_id, cx, cy, nw, nh))
    return labels


def _heatmap_split(stem: str) -> str:
    """Return 'train', 'val', or 'unknown' based on source image stem."""
    # Strip _k{N} suffix e.g. "page_23_k40" → "page_23"
    source_stem = re.sub(r"_k\d+$", "", stem)
    if (TRAIN_DIR / (source_stem + ".png")).exists() or (TRAIN_DIR / (source_stem + ".jpg")).exists():
        return "train"
    if (VALIDATE_DIR / (source_stem + ".png")).exists() or (VALIDATE_DIR / (source_stem + ".jpg")).exists():
        return "val"
    return "unknown"


def run_heatmaps_tab() -> None:
    """Heatmap generation and YOLO bounding-box labeling."""
    from PIL import Image as PILImage
    from streamlit_drawable_canvas import st_canvas

    # Ensure heatmap settings are in session_state
    settings = load_settings()
    for key, value in settings.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # ---- GENERATE SECTION ----
    st.subheader("Generate Heatmaps")

    train_images = get_image_files(TRAIN_DIR)
    val_images   = get_image_files(VALIDATE_DIR)
    n_rgb = len(list(HEATMAP_RGB_DIR.glob("*.png"))) if HEATMAP_RGB_DIR.exists() else 0

    met_cols = st.columns(3)
    met_cols[0].metric("Train images", len(train_images))
    met_cols[1].metric("Val images",   len(val_images))
    met_cols[2].metric("RGB heatmaps on disk", n_rgb)

    with st.expander("Generation Settings"):
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.slider("Kernel size (px)", 10, 100, key="heatmap_kernel", on_change=_on_setting_change)
            st.checkbox("Skip OCR (faster, G=0)", key="heatmap_skip_ocr", on_change=_on_setting_change)
        with g_col2:
            st.slider("Target hue (°)", 0, 360, key="heatmap_hue", on_change=_on_setting_change)
            st.slider("Target saturation (%)", 0, 100, key="heatmap_sat", on_change=_on_setting_change)
            st.slider("Target value/brightness (%)", 0, 100, key="heatmap_val", on_change=_on_setting_change)

    all_images = train_images + val_images
    if st.button("Generate Heatmaps", type="primary", disabled=not all_images):
        kernel = st.session_state.heatmap_kernel
        skip   = st.session_state.heatmap_skip_ocr
        hue    = st.session_state.heatmap_hue
        sat    = st.session_state.heatmap_sat / 100.0
        val    = st.session_state.heatmap_val / 100.0

        progress = st.progress(0)
        status   = st.empty()
        for i, img_path in enumerate(all_images):
            status.caption(f"Processing {img_path.name}...")
            _generate_heatmap(img_path, kernel, HEATMAP_OUTPUT_DIR, skip, hue, sat, val)
            progress.progress((i + 1) / len(all_images))
        progress.empty()
        status.empty()
        st.success(f"Generated {len(all_images)} heatmap(s) → `{HEATMAP_OUTPUT_DIR}/rgb/`")
        st.rerun()

    st.divider()

    # ---- LABEL SECTION ----
    st.subheader("Label Heatmaps")

    if not HEATMAP_RGB_DIR.exists() or not any(HEATMAP_RGB_DIR.iterdir()):
        st.info("No heatmaps yet — click **Generate Heatmaps** above.")
        return

    heatmap_files = sorted(
        p for p in HEATMAP_RGB_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    HEATMAP_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    n_labeled = sum(1 for f in HEATMAP_LABELS_DIR.glob("*.txt") if f.stat().st_size > 0)

    col_ctrl, col_canvas = st.columns([1, 4])

    with col_ctrl:
        st.metric("Labeled images", n_labeled, help="Images with at least one box saved")

        file_names = [f.name for f in heatmap_files]
        selected     = st.selectbox("Heatmap image", file_names, key="heatmap_file")
        heatmap_path = HEATMAP_RGB_DIR / selected
        label_path   = HEATMAP_LABELS_DIR / (heatmap_path.stem + ".txt")

        split = _heatmap_split(heatmap_path.stem)
        st.caption(f"Split: **{split}**")

        class_name = st.radio(
            "Draw class",
            options=[v["name"] for v in HEATMAP_CLASSES.values()],
            key="heatmap_class",
        )
        class_id     = _CLASS_BY_NAME[class_name]
        stroke_color = HEATMAP_CLASSES[class_id]["stroke"]
        fill_color   = HEATMAP_CLASSES[class_id]["fill"]

        existing_labels = load_yolo_labels(label_path)

        if st.button("Clear all boxes", key="heatmap_clear"):
            label_path.write_text("")
            st.rerun()

    with col_canvas:
        pil_img = PILImage.open(heatmap_path).convert("RGB")
        img_w, img_h = pil_img.size
        canvas_h = int(HEATMAP_CANVAS_WIDTH * img_h / img_w)

        initial_drawing = boxes_to_fabric(existing_labels, HEATMAP_CANVAS_WIDTH, canvas_h, img_w, img_h)

        # Form prevents reruns on every stroke — only fires on submit.
        with st.form("heatmap_label_form"):
            result = st_canvas(
                fill_color=fill_color,
                stroke_width=2,
                stroke_color=stroke_color,
                background_image=pil_img,
                initial_drawing=initial_drawing,
                update_streamlit=True,
                width=HEATMAP_CANVAS_WIDTH,
                height=canvas_h,
                drawing_mode="rect",
                key=f"canvas_{selected}",
            )

            submitted = st.form_submit_button("Save Labels", type="primary")

        if submitted and result.json_data is not None:
            labels = canvas_to_boxes(
                result.json_data, HEATMAP_CANVAS_WIDTH, canvas_h, img_w, img_h, class_id
            )
            save_yolo_labels(label_path, labels)
            st.success(f"Saved {len(labels)} box(es) → `{label_path}`")


def run_labeling_tab() -> None:
    """Run the labeling tab UI."""
    # Get available images
    image_files = get_image_files(TRAIN_DIR)

    if not image_files:
        st.error(f"No images found in {TRAIN_DIR}")
        st.info("Please add images to the train directory.")
        return

    # Load settings from disk and initialize session state
    settings = load_settings()
    for key, value in settings.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Controls row 1: Image selection and sensitivity
    col1, col2 = st.columns([2, 2])

    with col1:
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
            key="sensitivity",
            help="Higher = more sensitive (finds more features)",
            on_change=_on_setting_change,
        )

    # Controls row 2: Merge settings
    col3, col4 = st.columns([2, 2])

    with col3:
        merge_horizontal = st.slider(
            "Merge Horizontal Gap (px)",
            min_value=0,
            max_value=100,
            key="merge_horizontal",
            help="Merge boxes within this horizontal distance",
            on_change=_on_setting_change,
        )

    with col4:
        merge_vertical = st.slider(
            "Merge Vertical Gap (px)",
            min_value=0,
            max_value=100,
            key="merge_vertical",
            help="Merge boxes within this vertical distance",
            on_change=_on_setting_change,
        )

    # Load labels
    labels_data = load_labels(LABELS_FILE)
    existing_labels = get_labels_for_image(labels_data, selected_file)

    # Show label statistics
    all_labels = [entry["label"] for entry in labels_data["labels"]]
    label_counts = {
        "label": all_labels.count("label"),
        "figure": all_labels.count("figure"),
        "irrelevant": all_labels.count("irrelevant"),
    }
    total_labeled = sum(label_counts.values())

    stat_cols = st.columns(4)
    stat_cols[0].metric("Total Labeled", total_labeled)
    stat_cols[1].metric("Labels", label_counts["label"])
    stat_cols[2].metric("Figures", label_counts["figure"])
    stat_cols[3].metric("Irrelevant", label_counts["irrelevant"])

    # Run analysis
    with st.spinner("Extracting and analyzing features..."):
        annotated_image, results = run_analysis_pipeline(
            image_path, sensitivity, merge_horizontal, merge_vertical
        )

    st.success(f"Found {len(results)} features")

    # Show annotated image with numbered boxes
    st.image(annotated_image, use_column_width=True)

    st.divider()

    # Render results table with labeling inside a form
    with st.form("labeling_form"):
        render_table(results, selected_file, existing_labels)

        st.divider()
        submitted = st.form_submit_button("Save Labels", type="primary")

        if submitted:
            image_name = Path(selected_file).stem
            saved_count = 0
            for i, result in enumerate(results):
                key = f"label_{image_name}_{i}"
                selected_label = st.session_state.get(key)
                if selected_label:
                    add_label(labels_data, selected_file, i, selected_label, result)
                    saved_count += 1
            save_labels(labels_data, LABELS_FILE)
            st.success(f"Saved {saved_count} labels")


def run_train_and_validate() -> None:
    """Train classifier and run validation on all images."""
    # Train classifier
    classifier = train_classifier(LABELS_FILE)
    if classifier is None:
        st.error("Failed to train classifier - no labeled data found.")
        return

    st.success(f"Trained classifier on {classifier.n_samples} samples")

    # Get validation images
    validate_files = get_image_files(VALIDATE_DIR)
    if not validate_files:
        st.warning(f"No validation images found in {VALIDATE_DIR}")
        return

    # Process each validation image, keeping results grouped by source image
    results_by_image = []
    predictions_by_image = []

    progress = st.progress(0)
    for i, image_path in enumerate(validate_files):
        with st.spinner(f"Processing {image_path.name}..."):
            _, results = run_analysis_pipeline(image_path)
            if results:
                predictions = predict_features(results, classifier)
                results_by_image.append((image_path, results))
                predictions_by_image.append(predictions)
        progress.progress((i + 1) / len(validate_files))

    progress.empty()

    # Persist to disk
    save_validation_results(results_by_image, predictions_by_image, classifier.n_samples)

    # Flatten for session state
    all_results = [r for _, rs in results_by_image for r in rs]
    all_predictions = [p for ps in predictions_by_image for p in ps]
    st.session_state["validation_results"] = all_results
    st.session_state["validation_predictions"] = all_predictions


def run_validation_tab() -> None:
    """Run the validation tab UI."""
    # Restore from disk if session state is empty
    if "validation_results" not in st.session_state:
        loaded = load_validation_results()
        if loaded is not None:
            st.session_state["validation_results"], st.session_state["validation_predictions"] = loaded

    # Load labels to get count
    labels_data = load_labels(LABELS_FILE)
    n_labeled = len(labels_data.get("labels", []))

    # Get validation image count
    validate_files = get_image_files(VALIDATE_DIR)
    n_validate = len(validate_files)

    # Show statistics
    stat_cols = st.columns(2)
    stat_cols[0].metric("Labeled Samples", n_labeled)
    stat_cols[1].metric("Validation Images", n_validate)

    # Show warnings if prerequisites are missing
    if n_labeled == 0:
        st.warning("No labeled samples found. Please label some features in the Labeling tab first.")

    if n_validate == 0:
        st.warning(f"No validation images found in {VALIDATE_DIR}. Please add images to validate.")

    # Train button (disabled if prerequisites missing)
    can_train = n_labeled > 0 and n_validate > 0

    if st.button("Train Network and Check", type="primary", disabled=not can_train):
        run_train_and_validate()

    # Display results if available
    if "validation_results" in st.session_state and "validation_predictions" in st.session_state:
        results = st.session_state["validation_results"]
        predictions = st.session_state["validation_predictions"]

        if results:
            st.divider()
            st.subheader(f"Predictions ({len(results)} features)")
            render_prediction_table(results, predictions)


def run_app() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Image Feature Extraction",
        page_icon="?",
        layout="wide",
    )

    st.title("Image Feature Extraction & Analysis")

    # Create tabs
    labeling_tab, validation_tab, heatmap_tab = st.tabs(["Labeling", "Train & Validate", "Heatmaps"])

    with labeling_tab:
        run_labeling_tab()

    with validation_tab:
        run_validation_tab()

    with heatmap_tab:
        run_heatmaps_tab()
