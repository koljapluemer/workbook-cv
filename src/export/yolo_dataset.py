from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil


@dataclass(frozen=True)
class DatasetBuildResult:
    output_dir: Path
    train_count: int
    val_count: int
    test_count: int
    skipped_annotations: int


def _image_map(img_dir: Path) -> dict[str, Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    images = {}
    if not img_dir.exists():
        return images
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in extensions:
            images[p.stem] = p
    return images


def _copy_split(
    stems: list[str],
    img_map: dict[str, Path],
    annotations_dir: Path,
    images_out: Path,
    labels_out: Path,
) -> None:
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        img_path = img_map[stem]
        shutil.copy2(img_path, images_out / img_path.name)
        label_path = annotations_dir / f"{stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, labels_out / f"{stem}.txt")


def build_yolo_dataset(
    base_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetBuildResult:
    """
    Create a YOLO dataset from current project data.

    - Uses images in img/ that have annotations in processed/annotations/*.txt
    - Splits into train/val
    - Copies img/test-data into test (labels optional if they exist)
    """
    annotations_dir = base_dir / "processed" / "annotations"
    img_dir = base_dir / "img"
    test_dir = base_dir / "img" / "test-data"

    img_map = _image_map(img_dir)
    annotated_stems = [p.stem for p in annotations_dir.glob("*.txt")]
    valid_stems = [s for s in annotated_stems if s in img_map]
    skipped = len(annotated_stems) - len(valid_stems)

    rng = random.Random(seed)
    rng.shuffle(valid_stems)
    split_index = int(len(valid_stems) * train_ratio)
    train_stems = valid_stems[:split_index]
    val_stems = valid_stems[split_index:]

    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    images_test = output_dir / "images" / "test"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"
    labels_test = output_dir / "labels" / "test"

    _copy_split(train_stems, img_map, annotations_dir, images_train, labels_train)
    _copy_split(val_stems, img_map, annotations_dir, images_val, labels_val)

    test_count = 0
    if test_dir.exists():
        test_map = _image_map(test_dir)
        test_stems = list(test_map.keys())
        _copy_split(test_stems, test_map, annotations_dir, images_test, labels_test)
        test_count = len(test_stems)

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: datasets/wordbook",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: figure",
                "  1: figure_label",
                "",
            ]
        )
    )

    return DatasetBuildResult(
        output_dir=output_dir,
        train_count=len(train_stems),
        val_count=len(val_stems),
        test_count=test_count,
        skipped_annotations=skipped,
    )
