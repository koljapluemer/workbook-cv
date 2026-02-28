"""Generate pixel color diversity + text likelihood heatmaps for train images.

For each image in src/img/train/ produces an RGB heatmap where:
  R = local color diversity  (mean per-channel std dev in a sliding kernel)
  G = text likelihood        (sum of OCR confidence over 5 PSM modes per pixel)
  B = 0

Outputs per source image:
  data/heatmaps/heatmaps/{name}_k{N}.png   – heatmap only
  data/heatmaps/combined/{name}_k{N}.png   – original scan (left) | heatmap (right)

Usage:
    uv run python scripts/diversity_heatmap.py
    uv run python scripts/diversity_heatmap.py --kernel-size 64
    uv run python scripts/diversity_heatmap.py --skip-ocr
    uv run python scripts/diversity_heatmap.py --input-dir src/img/validate --output-dir data/validate_heatmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


TRAIN_DIR = Path("src/img/train")
DEFAULT_OUTPUT_DIR = Path("data/heatmaps")
DEFAULT_KERNEL_SIZE = 40

# Tesseract page segmentation modes used for text likelihood:
#   3  = fully automatic (default)
#   4  = single column, variable font sizes
#   6  = single uniform block of text
#   11 = sparse text, no particular order
#   12 = sparse text with OSD
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

def compute_text_likelihood(image: np.ndarray) -> np.ndarray:
    """Run OCR with multiple PSM modes; accumulate confidence per pixel.

    For each detected word box, adds confidence/100 to every pixel inside it.
    Running 5 PSM modes means a pixel solidly covered by text in all modes
    can reach a raw value of ~5.0 before normalisation.

    Returns:
        2D float32 array of accumulated confidence values.
    """
    try:
        import pytesseract
        from PIL import Image as PILImage
    except ImportError:
        raise RuntimeError(
            "pytesseract and Pillow are required for OCR. "
            "Install with: uv add pytesseract pillow"
        )

    h, w = image.shape[:2]
    likelihood = np.zeros((h, w), dtype=np.float32)
    pil_image = PILImage.fromarray(image)

    for psm in PSM_MODES:
        config = f"--psm {psm}"
        data = pytesseract.image_to_data(
            pil_image, config=config, output_type=pytesseract.Output.DICT
        )

        for i in range(len(data["text"])):
            if not data["text"][i].strip():
                continue
            bx = data["left"][i]
            by = data["top"][i]
            bw = data["width"][i]
            bh = data["height"][i]
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

    Each HSV component is normalised to [0, 1] before computing distance:
      - H: circular, max angular distance = 180° → normalised by 180
      - S: linear [0, 255]  → normalised by 255
      - V: linear [0, 255]  → normalised by 255

    The maximum possible distance in this normalised space is sqrt(3).
    Closeness = 1 - distance / sqrt(3), so 1 = exact match, 0 = opposite corner.

    Args:
        target_hue: Hue in degrees [0, 360).
        target_sat: Saturation in [0, 1].
        target_val: Value/brightness in [0, 1].
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[:, :, 0]   # OpenCV [0, 179]
    s = hsv[:, :, 1]   # [0, 255]
    v = hsv[:, :, 2]   # [0, 255]

    # Normalise hue distance (circular), max = 1
    target_h_cv = (target_hue % 360) / 2.0
    d_h = np.abs(h - target_h_cv)
    d_h = np.minimum(d_h, 180.0 - d_h) / 90.0   # [0, 1]

    # Normalise S and V distances, max = 1 each
    d_s = np.abs(s / 255.0 - target_sat)
    d_v = np.abs(v / 255.0 - target_val)

    distance = np.sqrt(d_h ** 2 + d_s ** 2 + d_v ** 2)   # max = sqrt(3)
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


def to_rgb_heatmap(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merge three uint8 channel arrays into a BGR image for cv2.imwrite."""
    return cv2.merge([b, g, r])


def process_image(
    image_path: Path,
    kernel_size: int,
    output_dir: Path,
    skip_ocr: bool,
    target_hue: int,
    target_sat: float,
    target_val: float,
) -> None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  Skipping {image_path.name} (could not read)")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    print(f"  {image_path.name} [{w}x{h}]", end="", flush=True)

    diversity = compute_diversity(image_rgb, kernel_size)
    print(" diversity", end="", flush=True)

    if skip_ocr:
        likelihood = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    else:
        likelihood = compute_text_likelihood(image_rgb)
        print(" OCR", end="", flush=True)

    hue_closeness = compute_color_closeness(image_rgb, target_hue, target_sat, target_val)
    print(" hue", end="", flush=True)

    r = norm_channel(diversity)
    g = norm_channel(likelihood)
    b = norm_channel(hue_closeness)

    heatmap_bgr = to_rgb_heatmap(r, g, b)
    stem = image_path.stem
    fname = f"{stem}_k{kernel_size}.png"

    cv2.imwrite(str(output_dir / "heatmaps" / fname), heatmap_bgr)
    cv2.imwrite(str(output_dir / "combined" / fname), np.concatenate([image_bgr, heatmap_bgr], axis=1))
    cv2.imwrite(str(output_dir / "channel_r" / fname), r)
    cv2.imwrite(str(output_dir / "channel_g" / fname), g)
    cv2.imwrite(str(output_dir / "channel_b" / fname), b)

    print(" -> saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pixel diversity + text likelihood heatmaps"
    )
    parser.add_argument(
        "--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE, metavar="N",
        help=f"Kernel side length in pixels (default: {DEFAULT_KERNEL_SIZE})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=TRAIN_DIR, metavar="DIR",
        help=f"Directory of input images (default: {TRAIN_DIR})",
    )
    parser.add_argument(
        "--skip-ocr", action="store_true",
        help="Skip OCR (G channel will be zero); useful for a quick diversity-only run",
    )
    parser.add_argument(
        "--target-hue", type=int, default=220, metavar="DEG",
        help="Target hue in degrees 0–360 for the B channel (default: 220)",
    )
    parser.add_argument(
        "--target-sat", type=float, default=0.65, metavar="S",
        help="Target saturation 0–1 for the B channel (default: 0.65)",
    )
    parser.add_argument(
        "--target-val", type=float, default=0.46, metavar="V",
        help="Target brightness/value 0–1 for the B channel (default: 0.46)",
    )
    args = parser.parse_args()

    for folder in ("heatmaps", "combined", "channel_r", "channel_g", "channel_b"):
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    )

    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    ocr_note = "skipped" if args.skip_ocr else f"{len(PSM_MODES)} PSM modes"
    print(f"Processing {len(image_paths)} image(s) | kernel={args.kernel_size}px | OCR={ocr_note} | target HSV=({args.target_hue}°, {args.target_sat:.0%}, {args.target_val:.0%})")
    print(f"Output: {args.output_dir}/\n")

    for image_path in image_paths:
        process_image(image_path, args.kernel_size, args.output_dir, args.skip_ocr, args.target_hue, args.target_sat, args.target_val)

    print("\nDone.")


if __name__ == "__main__":
    main()
