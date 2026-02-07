"""COCO JSON format export for annotations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def export_coco(
    images: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """
    Export annotations in COCO JSON format.

    Args:
        images: List of image dicts with keys:
            - file_name: str (image filename)
            - width: int
            - height: int
            - annotations: list of bbox dicts with keys:
                - bbox: [x, y, width, height]
                - category_id: int
        categories: List of category dicts with keys:
            - id: int
            - name: str
        output_path: Path to write the JSON file

    Returns:
        Path to the created file
    """
    # Build COCO structure
    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for img_idx, img in enumerate(images, start=1):
        # Add image entry
        coco_images.append({
            "id": img_idx,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        })

        # Add annotations for this image
        for ann in img.get("annotations", []):
            bbox = ann["bbox"]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": img_idx,
                "category_id": ann["category_id"],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            annotation_id += 1

    coco_data = {
        "info": {
            "description": "Wordbook-CV Annotations",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    return output_path
