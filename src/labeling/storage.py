"""JSON storage for feature labels."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.core import AnalysisResult


def load_labels(path: Path) -> dict:
    """Load labels from JSON file, creating empty structure if missing.

    Args:
        path: Path to the labels JSON file.

    Returns:
        Dictionary with version and labels list.
    """
    if path.exists():
        with open(path) as f:
            return json.load(f)

    return {"version": "1.0", "labels": []}


def save_labels(data: dict, path: Path) -> None:
    """Save labels to JSON file with atomic write.

    Args:
        data: Labels data dictionary.
        path: Path to save the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write using temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        suffix=".json",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = Path(tmp.name)

    tmp_path.replace(path)


def add_label(
    data: dict,
    image_file: str,
    feature_index: int,
    label: str,
    result: AnalysisResult,
) -> None:
    """Add or update a label entry in the data.

    Args:
        data: Labels data dictionary (modified in place).
        image_file: Name of the source image file.
        feature_index: Index of the feature in the image.
        label: Classification label ("label", "figure", or "irrelevant").
        result: AnalysisResult containing feature measurements.
    """
    # Generate ID
    image_name = Path(image_file).stem
    feature_id = f"{image_name}_{feature_index}"

    # Extract feature data
    h, w = result.feature.image.shape[:2]

    # Get raw histogram values from the feature image
    from src.analysis import compute_12bin_histogram

    hist_values = compute_12bin_histogram(result.feature.image)

    feature_data = {
        "text_coverage_percent": result.text_coverage_percent,
        "width": w,
        "height": h,
        "histogram": hist_values.tolist(),
    }

    # Build label entry
    entry = {
        "id": feature_id,
        "image_file": image_file,
        "feature_index": feature_index,
        "label": label,
        "features": feature_data,
    }

    # Update or append
    labels = data["labels"]
    for i, existing in enumerate(labels):
        if existing["id"] == feature_id:
            labels[i] = entry
            return

    labels.append(entry)


def get_labels_for_image(data: dict, image_file: str) -> dict[int, str]:
    """Get all labels for features in a specific image.

    Args:
        data: Labels data dictionary.
        image_file: Name of the image file.

    Returns:
        Dictionary mapping feature index to label string.
    """
    result = {}
    for entry in data["labels"]:
        if entry["image_file"] == image_file:
            result[entry["feature_index"]] = entry["label"]
    return result
