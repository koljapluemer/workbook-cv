from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class ImageVersion:
    """Represents a version of a processed image."""

    id: str  # "v001", "v002", etc.
    type: str  # "cleanup" | "bbox"
    source: str  # "original" | "v001" | etc.
    display_name: str  # For UI dropdown
    params: dict = field(default_factory=dict)  # Parameters used
    data: dict = field(default_factory=dict)  # Structured output (boxes, metrics)
    image: np.ndarray | None = None  # Loaded image (None if not loaded)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for storage."""
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "display_name": self.display_name,
            "created_at": self.created_at.isoformat(),
            "params": self.params,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict, image: np.ndarray | None = None) -> "ImageVersion":
        """Create ImageVersion from stored dict."""
        return cls(
            id=data["id"],
            type=data["type"],
            source=data["source"],
            display_name=data["display_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            params=data.get("params", {}),
            data=data.get("data", {}),
            image=image,
        )
