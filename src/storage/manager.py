import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .version import ImageVersion


class VersionManager:
    """Manages versioned images in processed/ directory."""

    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _get_version_dir(self, original: Path) -> Path:
        """Get the version directory for an original image."""
        return self.processed_dir / original.stem

    def _get_manifest_path(self, original: Path) -> Path:
        """Get the manifest.json path for an original image."""
        return self._get_version_dir(original) / "manifest.json"

    def _load_manifest(self, original: Path) -> dict:
        """Load manifest.json or return empty structure."""
        manifest_path = self._get_manifest_path(original)
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        return {"original": str(original), "versions": []}

    def _save_manifest(self, original: Path, manifest: dict) -> None:
        """Save manifest.json."""
        version_dir = self._get_version_dir(original)
        version_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._get_manifest_path(original)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _next_version_id(self, original: Path) -> str:
        """Get the next version ID (v001, v002, etc.)."""
        manifest = self._load_manifest(original)
        version_count = len(manifest["versions"])
        return f"v{version_count + 1:03d}"

    def create_version(
        self,
        original: Path,
        type: str,
        source: str,
        image: np.ndarray,
        params: dict,
        data: dict,
        display_name: str | None = None,
    ) -> ImageVersion:
        """Save image + data, update manifest, return ImageVersion."""
        version_dir = self._get_version_dir(original)
        version_dir.mkdir(parents=True, exist_ok=True)

        version_id = self._next_version_id(original)

        # Generate display name if not provided
        if display_name is None:
            if type == "cleanup":
                display_name = "Cleaned Up"
            elif type == "bbox":
                # Count existing bbox versions
                existing = self.list_versions(original)
                bbox_count = sum(1 for v in existing if v.type == "bbox") + 1
                display_name = f"Bounding Boxes #{bbox_count}"
            else:
                display_name = f"{type.title()} {version_id}"

        # Create version object
        version = ImageVersion(
            id=version_id,
            type=type,
            source=source,
            display_name=display_name,
            params=params,
            data=data,
            image=image,
        )

        # Save image as PNG (convert RGB to BGR for OpenCV)
        image_path = version_dir / f"{version_id}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save version data as JSON
        json_path = version_dir / f"{version_id}.json"
        with open(json_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        # Update manifest
        manifest = self._load_manifest(original)
        manifest["versions"].append(version_id)
        self._save_manifest(original, manifest)

        return version

    def get_version(self, original: Path, version_id: str) -> ImageVersion:
        """Load image + data for a version."""
        version_dir = self._get_version_dir(original)

        # Load JSON metadata
        json_path = version_dir / f"{version_id}.json"
        with open(json_path) as f:
            data = json.load(f)

        # Load image (convert BGR to RGB)
        image_path = version_dir / f"{version_id}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return ImageVersion.from_dict(data, image)

    def list_versions(self, original: Path) -> list[ImageVersion]:
        """Get all versions for an original image (without loading images)."""
        manifest = self._load_manifest(original)
        versions = []

        for version_id in manifest["versions"]:
            json_path = self._get_version_dir(original) / f"{version_id}.json"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                versions.append(ImageVersion.from_dict(data))

        return versions

    def load_original(self, original: Path) -> np.ndarray:
        """Load the original image as RGB numpy array."""
        image = Image.open(original)
        return np.array(image.convert("RGB"))
