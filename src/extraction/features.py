"""Feature extraction and temporary storage."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.core.types import BoundingBox, FeatureRect


def crop_feature(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Crop a feature from an image using a bounding box."""
    return image[box.y : box.y + box.h, box.x : box.x + box.w].copy()


def save_feature_to_temp(
    image: np.ndarray,
    index: int,
    temp_dir: Path,
) -> Path:
    """Save a feature image to the temp directory.

    Args:
        image: RGB image as numpy array.
        index: Feature index for filename.
        temp_dir: Directory to save to.

    Returns:
        Path to the saved file.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_path = temp_dir / f"feature_{index:04d}.png"

    # Convert RGB to BGR for cv2.imwrite
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)

    return output_path


def extract_all_features(
    image: np.ndarray,
    boxes: list[BoundingBox],
    temp_dir: Path,
) -> list[FeatureRect]:
    """Extract all features and save to temp directory.

    Args:
        image: RGB image as numpy array.
        boxes: List of bounding boxes to extract.
        temp_dir: Directory to save extracted features.

    Returns:
        List of FeatureRect objects.
    """
    clear_temp_directory(temp_dir)

    features = []
    for i, box in enumerate(boxes):
        cropped = crop_feature(image, box)
        temp_path = save_feature_to_temp(cropped, i, temp_dir)
        features.append(FeatureRect(box=box, image=cropped, temp_path=temp_path))

    return features


def clear_temp_directory(temp_dir: Path) -> None:
    """Clear all files from the temp directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
