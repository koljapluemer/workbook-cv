"""CSV export for ML training data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def export_to_csv(data: dict, path: Path) -> None:
    """Export labeled features to a flat CSV file.

    Columns: id, label, coverage, width, height, hist_r0..hist_b3

    Args:
        data: Labels data dictionary.
        path: Path to save the CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Column names
    histogram_cols = [
        "hist_r0",
        "hist_r1",
        "hist_r2",
        "hist_r3",
        "hist_g0",
        "hist_g1",
        "hist_g2",
        "hist_g3",
        "hist_b0",
        "hist_b1",
        "hist_b2",
        "hist_b3",
    ]
    fieldnames = ["id", "label", "coverage", "width", "height"] + histogram_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data["labels"]:
            features = entry["features"]
            histogram = features["histogram"]

            row = {
                "id": entry["id"],
                "label": entry["label"],
                "coverage": features["text_coverage_percent"],
                "width": features["width"],
                "height": features["height"],
            }

            # Add histogram values
            for i, col in enumerate(histogram_cols):
                row[col] = histogram[i] if i < len(histogram) else 0

            writer.writerow(row)


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV as numpy arrays for sklearn training.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (X features array, y labels array).
    """
    X_list = []
    y_list = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract features (coverage, width, height, histogram)
            features = [
                float(row["coverage"]),
                int(row["width"]),
                int(row["height"]),
            ]

            # Add histogram values
            for i in range(12):
                channel = ["r", "g", "b"][i // 4]
                bin_idx = i % 4
                col = f"hist_{channel}{bin_idx}"
                features.append(float(row[col]))

            X_list.append(features)
            y_list.append(row["label"])

    return np.array(X_list), np.array(y_list)
