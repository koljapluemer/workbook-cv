from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    label: str
    img_dir: Path
    read_only: bool


def get_image_files(img_dir: Path) -> list[Path]:
    """Get all image files from the directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    if not img_dir.exists():
        return []
    files = [f for f in img_dir.iterdir() if f.suffix.lower() in extensions]
    return sorted(files)


def load_datasets(base_dir: Path) -> list[DatasetConfig]:
    """Return available datasets based on folder structure."""
    datasets = [
        DatasetConfig(
            id="main",
            label="Main (img/)",
            img_dir=base_dir / "img",
            read_only=False,
        )
    ]

    test_dir = base_dir / "img" / "test-data"
    if get_image_files(test_dir):
        datasets.append(
            DatasetConfig(
                id="test",
                label="Test (img/test-data/)",
                img_dir=test_dir,
                read_only=True,
            )
        )

    return datasets


def get_dataset_by_id(datasets: list[DatasetConfig], dataset_id: str) -> DatasetConfig:
    """Find a dataset config by id, defaulting to the first."""
    for ds in datasets:
        if ds.id == dataset_id:
            return ds
    return datasets[0]
