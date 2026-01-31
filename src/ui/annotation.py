"""Annotation UI components for manual rectangle drawing."""

import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


def render_drawing_canvas(
    image: np.ndarray,
    existing_boxes: list[dict] | None = None,
    canvas_key: str = "annotation_canvas",
) -> dict | None:
    """
    Render a drawable canvas with the image as background.

    Args:
        image: RGB numpy array to display as background
        existing_boxes: Optional list of existing boxes to show (drawn but not editable)
        canvas_key: Unique key for the canvas component

    Returns:
        Canvas result object with json_data containing drawn rectangles
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Calculate canvas dimensions (maintain aspect ratio, max 800px width)
    max_width = 800
    if img_width > max_width:
        scale = max_width / img_width
        canvas_width = max_width
        canvas_height = int(img_height * scale)
    else:
        canvas_width = img_width
        canvas_height = img_height
        scale = 1.0

    # Draw existing boxes on the background image if provided
    if existing_boxes:
        display_image = image.copy()
        for box in existing_boxes:
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 200, 0), 2)
        pil_image = Image.fromarray(display_image)

    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=pil_image,
        drawing_mode="rect",
        width=canvas_width,
        height=canvas_height,
        key=canvas_key,
    )

    # Store scale factor in session for later use
    return {"canvas_result": canvas_result, "scale": scale}


def extract_boxes_from_canvas(
    canvas_data: dict,
    scale: float = 1.0,
) -> list[dict]:
    """
    Extract bounding boxes from canvas JSON data.

    Args:
        canvas_data: The canvas result containing json_data with drawn rectangles
        scale: Scale factor to convert canvas coordinates to original image coordinates

    Returns:
        List of box dicts with x, y, w, h keys (in original image coordinates)
    """
    boxes = []

    canvas_result = canvas_data.get("canvas_result")
    if canvas_result is None:
        return boxes

    json_data = canvas_result.json_data
    if json_data is None:
        return boxes

    objects = json_data.get("objects", [])

    for obj in objects:
        if obj.get("type") == "rect":
            # Get rectangle properties from fabric.js format
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            scale_x = obj.get("scaleX", 1)
            scale_y = obj.get("scaleY", 1)

            # Apply fabric.js scaling
            actual_width = width * scale_x
            actual_height = height * scale_y

            # Convert to original image coordinates
            inv_scale = 1.0 / scale if scale != 0 else 1.0
            box = {
                "x": int(left * inv_scale),
                "y": int(top * inv_scale),
                "w": int(actual_width * inv_scale),
                "h": int(actual_height * inv_scale),
            }

            # Only include boxes with positive dimensions
            if box["w"] > 0 and box["h"] > 0:
                boxes.append(box)

    return boxes


def render_annotated_image(
    source_image: np.ndarray,
    boxes: list[dict],
) -> np.ndarray:
    """
    Draw bounding boxes on the image.

    Args:
        source_image: RGB image to draw on
        boxes: List of box dicts with x, y, w, h keys

    Returns:
        Image with boxes drawn in green
    """
    result = source_image.copy()

    for i, box in enumerate(boxes):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        color = (0, 200, 0)

        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw box number
        label = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw background rectangle for text
        cv2.rectangle(
            result,
            (x, y - text_h - baseline - 4),
            (x + text_w + 4, y),
            color,
            -1,
        )

        # Draw text in white
        cv2.putText(
            result,
            label,
            (x + 2, y - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return result
