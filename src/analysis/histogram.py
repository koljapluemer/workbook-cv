"""12-bin color histogram computation and rendering."""

from __future__ import annotations

import numpy as np


def compute_12bin_histogram(image: np.ndarray) -> np.ndarray:
    """Compute a 12-bin color histogram (4 bins per RGB channel).

    Args:
        image: RGB image as numpy array.

    Returns:
        Array of 12 histogram values (R0-R3, G0-G3, B0-B3).
    """
    r_hist = np.histogram(image[:, :, 0], bins=4, range=(0, 256))[0]
    g_hist = np.histogram(image[:, :, 1], bins=4, range=(0, 256))[0]
    b_hist = np.histogram(image[:, :, 2], bins=4, range=(0, 256))[0]

    return np.concatenate([r_hist, g_hist, b_hist])


def render_histogram_image(
    histogram: np.ndarray,
    width: int = 240,
    height: int = 120,
) -> np.ndarray:
    """Render a histogram as a bar chart image.

    Args:
        histogram: Array of 12 histogram values.
        width: Output image width.
        height: Output image height.

    Returns:
        RGB image of the histogram bar chart.
    """
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    if len(histogram) != 12:
        return image

    # Normalize histogram for display
    max_val = histogram.max()
    if max_val == 0:
        max_val = 1

    normalized = histogram / max_val

    # Bar dimensions
    n_bars = 12
    bar_width = width // (n_bars + 1)
    padding = (width - bar_width * n_bars) // 2

    # Color shades for each channel (4 shades each: dark to light)
    red_shades = [(64, 0, 0), (128, 0, 0), (192, 0, 0), (255, 0, 0)]
    green_shades = [(0, 64, 0), (0, 128, 0), (0, 192, 0), (0, 255, 0)]
    blue_shades = [(0, 0, 64), (0, 0, 128), (0, 0, 192), (0, 0, 255)]
    colors = red_shades + green_shades + blue_shades

    # Draw bars
    for i, (value, color) in enumerate(zip(normalized, colors)):
        bar_height = int(value * (height - 20))
        if bar_height < 1:
            bar_height = 1

        x1 = padding + i * bar_width
        x2 = x1 + bar_width - 2
        y1 = height - 10 - bar_height
        y2 = height - 10

        # Fill bar
        image[y1:y2, x1:x2] = color

    return image
