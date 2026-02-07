from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


CATEGORIES = [
    {"id": 0, "name": "figure"},
    {"id": 1, "name": "figure_label"},
]


@dataclass(frozen=True)
class AutoAnnotateResult:
    annotated_count: int
    skipped_count: int
    output_dir: Path


def _write_coco_annotation(
    output_path: Path,
    image_name: str,
    width: int,
    height: int,
    boxes: list[dict],
) -> None:
    data = {
        "image": {"file_name": image_name, "width": width, "height": height},
        "annotations": [],
        "categories": CATEGORIES,
    }

    for i, ann in enumerate(boxes):
        data["annotations"].append(
            {
                "id": i,
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
            }
        )

    output_path.write_text(_json_dumps(data))


def _write_yolo_annotation(
    output_path: Path,
    width: int,
    height: int,
    boxes: list[dict],
) -> None:
    lines: list[str] = []
    for ann in boxes:
        x, y, w, h = ann["bbox"]
        center_x = (x + w / 2) / width
        center_y = (y + h / 2) / height
        norm_w = w / width
        norm_h = h / height
        lines.append(
            f"{ann['category_id']} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
        )
    output_path.write_text("\n".join(lines))


def _json_dumps(data: dict) -> str:
    import json

    return json.dumps(data, indent=2)


def _image_files(img_dir: Path) -> Iterable[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    if not img_dir.exists():
        return []
    return [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions]


def train_yolo_model(
    base_dir: Path,
    data_dir: Path,
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
) -> Path:
    from ultralytics import YOLO

    project_dir = base_dir / "runs"
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    result = model.train(
        data=str(data_dir / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project_dir),
        name="wordbook",
        exist_ok=True,
        verbose=False,
    )
    return Path(result.save_dir) / "weights" / "best.pt"


def auto_annotate_test_images(
    base_dir: Path,
    weights_path: Path,
    conf: float = 0.25,
) -> AutoAnnotateResult:
    from ultralytics import YOLO

    test_dir = base_dir / "img" / "test-data"
    annotations_dir = base_dir / "processed" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    annotated = 0
    skipped = 0
    for image_path in _image_files(test_dir):
        with Image.open(image_path) as img:
            width, height = img.size

        results = model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
            save=False,
        )
        if not results:
            skipped += 1
            continue

        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0].item())
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "category_id": cls_id,
                    }
                )

        stem = image_path.stem
        json_path = annotations_dir / f"{stem}.json"
        txt_path = annotations_dir / f"{stem}.txt"
        _write_coco_annotation(json_path, image_path.name, width, height, boxes)
        _write_yolo_annotation(txt_path, width, height, boxes)
        annotated += 1

    return AutoAnnotateResult(
        annotated_count=annotated,
        skipped_count=skipped,
        output_dir=annotations_dir,
    )
