import cv2
import numpy as np


def detect_bounding_boxes(
    image: np.ndarray, sensitivity: int = 5
) -> tuple[np.ndarray, list[dict]]:
    """
    Detect text blocks and graphics in a textbook scan using contour detection.

    Args:
        image: RGB image as numpy array
        sensitivity: Detection sensitivity 1-10 (higher = more boxes)

    Returns:
        Tuple of (image_with_boxes, list_of_box_dicts)
    """
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adjust adaptive threshold block size based on sensitivity
    block_size = 21 - sensitivity  # 11-20 based on sensitivity 1-10
    if block_size % 2 == 0:
        block_size += 1  # Must be odd
    block_size = max(3, block_size)

    # Use adaptive thresholding for better results on scanned documents
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2
    )

    # Dilate to connect nearby text/elements into blocks
    # Adjust kernel size based on sensitivity
    kernel_w = 20 - sensitivity  # 10-19 based on sensitivity
    kernel_h = 8 - (sensitivity // 2)  # 3-8 based on sensitivity
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, kernel_w), max(2, kernel_h)))
    dilated = cv2.dilate(binary, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw bounding boxes
    img_height, img_width = image.shape[:2]

    # Adjust area thresholds based on sensitivity
    min_area_pct = 0.002 - (sensitivity * 0.0001)  # 0.1%-0.2% based on sensitivity
    min_area = (img_width * img_height) * max(0.0005, min_area_pct)
    max_area = (img_width * img_height) * 0.95  # Max 95% of image area

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Filter by area and aspect ratio
        if min_area < area < max_area:
            # Draw red bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

    # Sort boxes by y position (top to bottom), then x (left to right)
    boxes.sort(key=lambda b: (b["y"], b["x"]))

    return result, boxes
