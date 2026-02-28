"""Heatmap generation: pixel color diversity, text likelihood, and HSV color closeness."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PSM_MODES = [3, 4, 6, 11, 12]


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def compute_diversity(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Local color diversity as mean per-channel std dev via box filters.

    Uses Var(X) = E[X²] - E[X]² — O(n) regardless of kernel size.

    Returns:
        2D float32 array, values roughly in [0, 255].
    """
    k = (kernel_size, kernel_size)
    image_f = image.astype(np.float32)
    diversity = np.zeros(image_f.shape[:2], dtype=np.float32)

    for c in range(3):
        ch = image_f[:, :, c]
        mean = cv2.boxFilter(ch, ddepth=-1, ksize=k)
        mean_sq = cv2.boxFilter(ch * ch, ddepth=-1, ksize=k)
        variance = np.maximum(mean_sq - mean * mean, 0.0)
        diversity += np.sqrt(variance)

    return diversity / 3.0


# ---------------------------------------------------------------------------
# Text likelihood
# ---------------------------------------------------------------------------

def _make_binarized(image: np.ndarray) -> np.ndarray:
    """Return an adaptive-binarized grayscale version of an RGB image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=10
    )


def _accumulate_ocr(pil_image, likelihood: np.ndarray, h: int, w: int, psm: int) -> None:
    """Run one Tesseract pass and add confidence values into likelihood in-place."""
    import pytesseract
    data = pytesseract.image_to_data(
        pil_image, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    for i in range(len(data["text"])):
        if not data["text"][i].strip():
            continue
        bx, by, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if bw <= 0 or bh <= 0:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            continue
        if conf < 0:
            continue
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w, bx + bw), min(h, by + bh)
        likelihood[y1:y2, x1:x2] += conf / 100.0


def compute_text_likelihood(image: np.ndarray) -> np.ndarray:
    """Run OCR with multiple PSM modes on both the raw and binarized image.

    Returns:
        2D float32 array of accumulated confidence values.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise RuntimeError("Pillow is required. Install with: uv add pillow")

    h, w = image.shape[:2]
    likelihood = np.zeros((h, w), dtype=np.float32)

    pil_raw = PILImage.fromarray(image)
    pil_bin = PILImage.fromarray(_make_binarized(image))

    for psm in PSM_MODES:
        _accumulate_ocr(pil_raw, likelihood, h, w, psm)
        _accumulate_ocr(pil_bin, likelihood, h, w, psm)

    return likelihood


# ---------------------------------------------------------------------------
# HSV color closeness
# ---------------------------------------------------------------------------

def compute_color_closeness(
    image: np.ndarray,
    target_hue: int = 220,
    target_sat: float = 0.65,
    target_val: float = 0.46,
) -> np.ndarray:
    """Per-pixel Euclidean closeness to a target HSV color.

    Args:
        target_hue: Hue in degrees [0, 360).
        target_sat: Saturation in [0, 1].
        target_val: Value/brightness in [0, 1].
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[:, :, 0]   # OpenCV [0, 179]
    s = hsv[:, :, 1]   # [0, 255]
    v = hsv[:, :, 2]   # [0, 255]

    target_h_cv = (target_hue % 360) / 2.0
    d_h = np.abs(h - target_h_cv)
    d_h = np.minimum(d_h, 180.0 - d_h) / 90.0

    d_s = np.abs(s / 255.0 - target_sat)
    d_v = np.abs(v / 255.0 - target_val)

    distance = np.sqrt(d_h ** 2 + d_s ** 2 + d_v ** 2)
    closeness = 1.0 - distance / np.sqrt(3.0)
    return closeness.astype(np.float32)


# ---------------------------------------------------------------------------
# Combining + saving
# ---------------------------------------------------------------------------

def norm_channel(arr: np.ndarray) -> np.ndarray:
    """Normalise a float32 map to uint8 [0, 255]."""
    if arr.max() == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def process_image(
    image_path: Path,
    kernel_size: int,
    output_dir: Path,
    skip_ocr: bool,
    target_hue: int,
    target_sat: float,
    target_val: float,
) -> None:
    """Generate all heatmap outputs for a single source image.

    Outputs go into subdirectories of output_dir:
      rgb/        — full RGB heatmap (R=diversity, G=text, B=color closeness)
      combined/   — original scan (left) | heatmap (right)
      channel_r/  — R channel only
      channel_g/  — G channel only
      channel_b/  — B channel only
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    diversity = compute_diversity(image_rgb, kernel_size)

    if skip_ocr:
        likelihood = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    else:
        likelihood = compute_text_likelihood(image_rgb)

    hue_closeness = compute_color_closeness(image_rgb, target_hue, target_sat, target_val)

    r = norm_channel(diversity)
    g = norm_channel(likelihood)
    b = norm_channel(hue_closeness)

    # BGR for cv2.imwrite (channels: B=blue channel, G=green, R=red)
    heatmap_bgr = cv2.merge([b, g, r])
    stem = image_path.stem
    fname = f"{stem}_k{kernel_size}.png"

    for subdir in ("rgb", "combined", "channel_r", "channel_g", "channel_b"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "rgb" / fname), heatmap_bgr)
    cv2.imwrite(str(output_dir / "combined" / fname), np.concatenate([image_bgr, heatmap_bgr], axis=1))
    cv2.imwrite(str(output_dir / "channel_r" / fname), r)
    cv2.imwrite(str(output_dir / "channel_g" / fname), g)
    cv2.imwrite(str(output_dir / "channel_b" / fname), b)
