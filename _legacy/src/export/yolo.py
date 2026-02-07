"""YOLO format export for annotations."""

from pathlib import Path
from typing import Any


def export_yolo(
    images: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Export annotations in YOLO txt format.

    Creates:
        - labels/*.txt files (one per image)
        - classes.txt file

    Args:
        images: List of image dicts with keys:
            - file_name: str (image filename)
            - width: int
            - height: int
            - annotations: list of bbox dicts with keys:
                - bbox: [x, y, width, height] (absolute pixels)
                - category_id: int
        categories: List of category dicts with keys:
            - id: int
            - name: str
        output_dir: Directory to write files

    Returns:
        Path to the output directory

    YOLO format per line: class_id x_center y_center width height
    All values normalized to 0-1 range.
    """
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Write classes.txt
    # Sort categories by id to ensure consistent ordering
    sorted_categories = sorted(categories, key=lambda c: c["id"])
    classes_path = output_dir / "classes.txt"
    with open(classes_path, "w") as f:
        for cat in sorted_categories:
            f.write(f"{cat['name']}\n")

    # Create a mapping from category_id to class index (0-based)
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(sorted_categories)}

    # Write label files
    for img in images:
        img_width = img["width"]
        img_height = img["height"]

        # Create label filename (same name as image, but .txt)
        img_stem = Path(img["file_name"]).stem
        label_path = labels_dir / f"{img_stem}.txt"

        lines = []
        for ann in img.get("annotations", []):
            bbox = ann["bbox"]  # [x, y, w, h] in pixels
            x, y, w, h = bbox

            # Convert to YOLO format (center x, center y, width, height) normalized
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height

            class_idx = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    return output_dir
