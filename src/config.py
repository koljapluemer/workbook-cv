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

# Validation persistence
VALIDATION_STORE_DIR = Path("data/validation")

# Heatmaps
HEATMAP_OUTPUT_DIR    = Path("data/heatmaps")
HEATMAP_RGB_DIR       = Path("data/heatmaps/rgb")
HEATMAP_COMBINED_DIR  = Path("data/heatmaps/combined")
HEATMAP_CHANNEL_R_DIR = Path("data/heatmaps/channel_r")
HEATMAP_CHANNEL_G_DIR = Path("data/heatmaps/channel_g")
HEATMAP_CHANNEL_B_DIR = Path("data/heatmaps/channel_b")
HEATMAP_LABELS_DIR    = Path("data/heatmap_labels")
HEATMAP_SEG_DIR       = Path("data/heatmap_seg")
HEATMAP_MODEL_PATH    = Path("data/models/heatmap_detector/weights/best.pt")
HEATMAP_DETECTIONS_DIR = Path("data/heatmap_detections")
FLASHCARD_EXPORT_DIR  = Path("data/flashcard_export")

HEATMAP_DEFAULT_KERNEL = 30
HEATMAP_DEFAULT_HUE    = 220
HEATMAP_DEFAULT_SAT    = 65   # 0-100 integer (percent)
HEATMAP_DEFAULT_VAL    = 46   # 0-100 integer (percent)
