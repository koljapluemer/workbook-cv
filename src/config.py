"""Configuration constants for the image feature extraction app."""

from pathlib import Path

# Directory paths
TRAIN_DIR = Path("src/img/train")
TEMP_DIR = Path("temp")

# OCR overlay settings
OCR_OVERLAY_COLOR = (255, 215, 0)  # Yellow in BGR

# Detection settings
DEFAULT_SENSITIVITY = 8

# Labeling
LABELS_FILE = Path("data/labels.json")

# Validation
VALIDATE_DIR = Path("src/img/validate")

# UI Settings
SETTINGS_FILE = Path("data/settings.json")
