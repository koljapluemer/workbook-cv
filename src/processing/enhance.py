import cv2
import numpy as np


def denoise(image: np.ndarray, strength: int = 10) -> tuple[np.ndarray, dict]:
    """
    Apply denoising using fastNlMeansDenoisingColored.

    Args:
        image: RGB image
        strength: Filter strength (higher = more smoothing)

    Returns:
        Tuple of (denoised_image, metadata_dict)
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, strength, strength, 7, 21
    )
    return denoised, {"denoise_strength": strength}


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> tuple[np.ndarray, dict]:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB L-channel.

    Args:
        image: RGB image
        clip_limit: Threshold for contrast limiting

    Returns:
        Tuple of (enhanced_image, metadata_dict)
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return enhanced, {"clahe_clip_limit": clip_limit}


def white_balance(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Apply gray-world white balance normalization.

    Returns:
        Tuple of (balanced_image, metadata_dict)
    """
    result = image.copy().astype(np.float32)

    # Calculate average of each channel
    avg_b = np.mean(result[:, :, 2])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 0])

    # Calculate overall average
    avg = (avg_b + avg_g + avg_r) / 3

    # Scale each channel
    if avg_b > 0:
        result[:, :, 2] *= avg / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg / avg_g
    if avg_r > 0:
        result[:, :, 0] *= avg / avg_r

    # Clip to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result, {
        "channel_scales": {
            "r": float(avg / avg_r) if avg_r > 0 else 1.0,
            "g": float(avg / avg_g) if avg_g > 0 else 1.0,
            "b": float(avg / avg_b) if avg_b > 0 else 1.0,
        }
    }
