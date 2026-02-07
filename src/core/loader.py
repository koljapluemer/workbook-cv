"""Image loading utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def get_image_files(directory: Path) -> list[Path]:
    """List all supported image files in a directory, sorted alphabetically."""
    if not directory.exists():
        return []

    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def load_first_image(directory: Path) -> tuple[np.ndarray, Path] | None:
    """Load the first image from a directory (sorted alphabetically).

    Returns:
        Tuple of (image as RGB numpy array, path) or None if no images found.
    """
    files = get_image_files(directory)
    if not files:
        return None

    image_path = files[0]
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, image_path
