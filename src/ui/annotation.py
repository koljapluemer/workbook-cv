"""Annotation UI components for manual rectangle drawing."""

import json
import math
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Category definitions with colors
CATEGORIES = {
    "figure": {"color_rgb": (0, 200, 0), "color_hex": "#00C800", "fill": "rgba(0, 200, 0, 0.3)", "id": 0},
    "figure_label": {"color_rgb": (0, 100, 255), "color_hex": "#0064FF", "fill": "rgba(0, 100, 255, 0.3)", "id": 1},
}

# Global storage path and image dimensions cache
_storage_path: Path | None = None
_image_dims: dict[str, tuple[int, int]] = {}  # image_name -> (width, height)


def init_annotation_storage(processed_dir: Path) -> None:
    """Initialize the annotation storage directory."""
    global _storage_path
    _storage_path = processed_dir / "annotations"
    _storage_path.mkdir(parents=True, exist_ok=True)


def set_image_dimensions(image_name: str, width: int, height: int) -> None:
    """Cache image dimensions for YOLO normalization."""
    _image_dims[image_name] = (width, height)


def _get_coco_path(image_name: str) -> Path:
    """Get COCO JSON path for an image."""
    return _storage_path / f"{image_name}.json"


def _get_yolo_path(image_name: str) -> Path:
    """Get YOLO txt path for an image."""
    return _storage_path / f"{image_name}.txt"


def _load_from_coco(image_name: str) -> list[dict]:
    """Load annotations from COCO JSON (source of truth)."""
    path = _get_coco_path(image_name)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)

    annotations = []
    for ann in data.get("annotations", []):
        bbox = ann["bbox"]  # [x, y, width, height]
        cat_id = ann["category_id"]
        # Find category name by id
        cat_name = "figure"
        for name, info in CATEGORIES.items():
            if info["id"] == cat_id:
                cat_name = name
                break
        annotations.append({
            "box": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
            "category": cat_name,
        })
    return annotations


def _save_to_coco(image_name: str, annotations: list[dict]) -> None:
    """Save annotations to COCO JSON format."""
    dims = _image_dims.get(image_name, (0, 0))

    coco_data = {
        "image": {
            "file_name": f"{image_name}.jpg",
            "width": dims[0],
            "height": dims[1],
        },
        "annotations": [],
        "categories": [{"id": info["id"], "name": name} for name, info in CATEGORIES.items()],
    }

    for i, ann in enumerate(annotations):
        box = ann["box"]
        cat_name = ann["category"]
        cat_id = CATEGORIES.get(cat_name, CATEGORIES["figure"])["id"]
        coco_data["annotations"].append({
            "id": i,
            "bbox": [box["x"], box["y"], box["w"], box["h"]],
            "category_id": cat_id,
        })

    with open(_get_coco_path(image_name), "w") as f:
        json.dump(coco_data, f, indent=2)


def _save_to_yolo(image_name: str, annotations: list[dict]) -> None:
    """Save annotations to YOLO format (normalized coordinates)."""
    dims = _image_dims.get(image_name)
    if not dims or dims[0] == 0 or dims[1] == 0:
        return

    img_w, img_h = dims
    lines = []

    for ann in annotations:
        box = ann["box"]
        cat_name = ann["category"]
        cat_id = CATEGORIES.get(cat_name, CATEGORIES["figure"])["id"]

        # Convert to YOLO format: center_x, center_y, width, height (normalized)
        center_x = (box["x"] + box["w"] / 2) / img_w
        center_y = (box["y"] + box["h"] / 2) / img_h
        norm_w = box["w"] / img_w
        norm_h = box["h"] / img_h

        lines.append(f"{cat_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    with open(_get_yolo_path(image_name), "w") as f:
        f.write("\n".join(lines))


def _save_annotations(image_name: str, annotations: list[dict]) -> None:
    """Save annotations to both COCO and YOLO formats."""
    _save_to_coco(image_name, annotations)
    _save_to_yolo(image_name, annotations)


def get_annotations_key(image_name: str) -> str:
    """Get session state key for annotations."""
    return f"annotations_{image_name}"


def get_annotations(image_name: str) -> list[dict]:
    """Get annotations from session state, loading from COCO if needed."""
    key = get_annotations_key(image_name)
    if key not in st.session_state:
        st.session_state[key] = _load_from_coco(image_name)
    return st.session_state[key]


def add_annotation(image_name: str, box: dict, category: str) -> None:
    """Add a new annotation and persist."""
    annotations = get_annotations(image_name)
    annotations.append({"box": box, "category": category})
    st.session_state[get_annotations_key(image_name)] = annotations
    _save_annotations(image_name, annotations)


def delete_annotation(image_name: str, index: int) -> None:
    """Delete an annotation by index and persist."""
    annotations = get_annotations(image_name)
    if 0 <= index < len(annotations):
        annotations.pop(index)
        st.session_state[get_annotations_key(image_name)] = annotations
        _save_annotations(image_name, annotations)
        # Increment canvas reset counter to force canvas refresh
        reset_key = f"canvas_reset_{image_name}"
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1


def render_category_toggle() -> str:
    """Render category selection toggle, returns selected category."""
    st.subheader("Draw Annotations")
    category = st.radio(
        "Category",
        options=list(CATEGORIES.keys()),
        horizontal=True,
        key="annotation_category",
    )
    return category


def render_annotation_list(image_name: str, source_image: np.ndarray, read_only: bool = False) -> bool:
    """
    Render list of annotations with thumbnails and delete buttons.
    Returns True if any annotation was deleted.
    """
    annotations = get_annotations(image_name)

    if not annotations:
        st.caption("No annotations yet. Draw rectangles on the image.")
        return False

    st.caption(f"{len(annotations)} annotation(s)")

    deleted = False
    for i, ann in enumerate(annotations):
        box = ann["box"]
        category = ann["category"]
        cat_info = CATEGORIES.get(category, CATEGORIES["figure"])

        # Extract thumbnail
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        img_h, img_w = source_image.shape[:2]
        x1 = int(max(0, min(x, img_w)))
        y1 = int(max(0, min(y, img_h)))
        x2 = int(max(0, min(x + w, img_w)))
        y2 = int(max(0, min(y + h, img_h)))

        if x2 > x1 and y2 > y1:
            thumbnail = source_image[y1:y2, x1:x2]
            thumb_h = min(60, thumbnail.shape[0])
            thumb_w = int(thumbnail.shape[1] * (thumb_h / thumbnail.shape[0]))
            if thumb_w > 0:
                thumbnail = cv2.resize(thumbnail, (thumb_w, thumb_h))
            else:
                thumbnail = None
        else:
            thumbnail = None

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if thumbnail is not None:
                st.image(thumbnail, use_column_width=True)

        with col2:
            color_hex = cat_info["color_hex"]
            st.markdown(f"<span style='color:{color_hex}'>{category}</span>", unsafe_allow_html=True)
            st.caption(f"{w}x{h}")

        with col3:
            if read_only:
                st.button("X", key=f"del_{image_name}_{i}", disabled=True)
            else:
                if st.button("X", key=f"del_{image_name}_{i}"):
                    delete_annotation(image_name, i)
                    deleted = True

    return deleted


def render_drawing_canvas(
    image: np.ndarray,
    image_name: str,
    category: str,
    canvas_key: str = "annotation_canvas",
    read_only: bool = False,
) -> None:
    """
    Render a drawable canvas with the image as background.
    Auto-persists new rectangles.
    """
    img_height, img_width = image.shape[:2]
    set_image_dimensions(image_name, img_width, img_height)

    # Include reset counter in key to force refresh after deletions
    reset_count = st.session_state.get(f"canvas_reset_{image_name}", 0)
    canvas_key = f"{canvas_key}_{reset_count}"

    # Calculate canvas dimensions (max 800px width)
    max_width = 800
    if img_width > max_width:
        scale = max_width / img_width
        canvas_width = max_width
        canvas_height = int(img_height * scale)
    else:
        canvas_width = img_width
        canvas_height = img_height
        scale = 1.0

    # Draw existing annotations on the background
    annotations = get_annotations(image_name)
    display_image = image.copy()
    for ann in annotations:
        box = ann["box"]
        cat = ann["category"]
        color = CATEGORIES.get(cat, CATEGORIES["figure"])["color_rgb"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, w, h)):
            continue
        x_i = int(round(x))
        y_i = int(round(y))
        w_i = int(round(w))
        h_i = int(round(h))
        if w_i <= 0 or h_i <= 0:
            continue
        cv2.rectangle(display_image, (x_i, y_i), (x_i + w_i, y_i + h_i), color, 2)
        cv2.putText(
            display_image,
            cat[:3],
            (x_i + 2, y_i + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    pil_image = Image.fromarray(display_image)
    cat_info = CATEGORIES.get(category, CATEGORIES["figure"])

    if read_only:
        st.image(pil_image, width=canvas_width)
        return

    canvas_result = st_canvas(
        fill_color=cat_info["fill"],
        stroke_width=2,
        stroke_color=cat_info["color_hex"],
        background_image=pil_image,
        drawing_mode="rect",
        width=canvas_width,
        height=canvas_height,
        key=canvas_key,
    )

    # Auto-persist new rectangles (no rerun to avoid interrupting drawing)
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        new_boxes = _extract_boxes(objects, scale)

        canvas_count_key = f"{canvas_key}_count"
        prev_count = st.session_state.get(canvas_count_key, 0)

        if len(new_boxes) > prev_count:
            for box in new_boxes[prev_count:]:
                add_annotation(image_name, box, category)
            st.session_state[canvas_count_key] = len(new_boxes)


def _extract_boxes(objects: list, scale: float) -> list[dict]:
    """Extract box dicts from canvas objects."""
    boxes = []
    for obj in objects:
        if obj.get("type") == "rect":
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            scale_x = obj.get("scaleX", 1)
            scale_y = obj.get("scaleY", 1)

            actual_width = width * scale_x
            actual_height = height * scale_y
            inv_scale = 1.0 / scale if scale != 0 else 1.0

            box = {
                "x": int(left * inv_scale),
                "y": int(top * inv_scale),
                "w": int(actual_width * inv_scale),
                "h": int(actual_height * inv_scale),
            }

            if box["w"] > 0 and box["h"] > 0:
                boxes.append(box)
    return boxes
