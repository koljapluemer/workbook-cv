from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def detect_ocr(image: np.ndarray) -> list[dict[str, Any]]:
    """
    Run OCR and return a list of word boxes.

    Each item:
        {
            "text": str,
            "bbox": [x, y, w, h],
            "conf": float
        }
    """
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pytesseract is not installed.") from exc

    pil_image = Image.fromarray(image)
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

    results: list[dict[str, Any]] = []
    count = len(data.get("text", []))
    for i in range(count):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf_raw = data.get("conf", [])[i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        results.append(
            {
                "text": text,
                "bbox": [x, y, w, h],
                "conf": conf,
            }
        )

    return results
