"""Heatmap-based region detector — segmentation backend.

Architecture:
    HeatmapDetector (ABC)
        └── SegmentationDetector   ← U-Net trained from scratch on RGB heatmaps

Adding option 3 (patch classifier) later:
    Implement PatchClassifierDetector(HeatmapDetector) in this file.
    Swap it into _run_seg_training / _run_seg_detection in app.py.
    Zero UI changes required.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class HeatmapDetector(ABC):
    CLASS_NAMES  = {0: "drawing", 1: "textlabel"}
    CLASS_COLORS = {0: (68, 68, 255), 1: (255, 136, 68)}  # BGR

    @abstractmethod
    def train(self, image_paths: list[Path], label_paths: list[Path], epochs: int) -> None:
        """Train on labeled heatmaps."""

    @abstractmethod
    def detect(
        self,
        image_path: Path,
        threshold: float = 0.5,
        debug_dir: Path | None = None,
    ) -> list[tuple[int, float, float, float, float]]:
        """Return list of (class_id, x1n, y1n, x2n, y2n) normalized to [0, 1]."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model weights to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "HeatmapDetector":
        """Load a previously saved model from disk."""


# ---------------------------------------------------------------------------
# Shared label utilities
# ---------------------------------------------------------------------------

def yolo_labels_to_mask(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """Convert a YOLO .txt label file to a uint8 segmentation mask.

    Mask values: 0 = background, 1 = drawing, 2 = textlabel.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return mask
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, nw, nh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - nw / 2) * img_w)
        y1 = int((cy - nh / 2) * img_h)
        x2 = int((cx + nw / 2) * img_w)
        y2 = int((cy + nh / 2) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        mask[y1:y2, x1:x2] = cls + 1  # 0→1 (drawing), 1→2 (textlabel)
    return mask


def mask_to_boxes(
    mask: np.ndarray, class_idx: int, min_area: int = 500
) -> list[tuple[float, float, float, float]]:
    """Extract normalized (x1, y1, x2, y2) bounding boxes from a binary class mask.

    Args:
        mask: 2-D uint8 array where non-zero = foreground.
        class_idx: unused here but kept for interface clarity.
        min_area: minimum pixel area to keep a component.

    Returns:
        List of (x1n, y1n, x2n, y2n) in [0, 1].
    """
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    boxes = []
    for i in range(1, n_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x / w, y / h, (x + bw) / w, (y + bh) / h))
    return boxes


# ---------------------------------------------------------------------------
# U-Net model
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UNet(nn.Module):
    """Lightweight U-Net (~1 M params) trained from scratch on heatmap data."""

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        # Encoder
        self.enc1 = _ConvBlock(3, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.enc3 = _ConvBlock(64, 128)
        self.enc4 = _ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = _ConvBlock(256, 512)
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _ConvBlock(64, 32)
        # Head
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _HeatmapDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        label_paths: list[Path],
        img_size: int,
        augment: bool = True,
    ) -> None:
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        mask = yolo_labels_to_mask(self.label_paths[idx], h, w)

        # Resize to training size
        s = self.img_size
        img  = cv2.resize(img,  (s, s), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        # Augmentation: random flips + 90° rotations
        if self.augment:
            if random.random() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            k = random.randint(0, 3)
            if k:
                img  = np.rot90(img,  k).copy()
                mask = np.rot90(mask, k).copy()

        img_t  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


# ---------------------------------------------------------------------------
# SegmentationDetector
# ---------------------------------------------------------------------------

class SegmentationDetector(HeatmapDetector):
    """U-Net trained from scratch on RGB heatmaps for region detection."""

    IMG_SIZE   = 512
    N_CLASSES  = 3      # background, drawing, textlabel
    BATCH_SIZE = 2

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = _UNet(self.N_CLASSES).to(self.device)

    # ------------------------------------------------------------------
    def train(
        self,
        image_paths: list[Path],
        label_paths: list[Path],
        epochs: int,
    ) -> None:
        dataset = _HeatmapDataset(image_paths, label_paths, self.IMG_SIZE, augment=True)
        loader  = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0)

        # Down-weight background class so object pixels dominate the loss
        weights = torch.tensor([0.1, 1.0, 1.0], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, masks in loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                logits = self.model(imgs)
                loss   = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(len(loader), 1)
            scheduler.step(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

    # ------------------------------------------------------------------
    def detect(
        self,
        image_path: Path,
        threshold: float = 0.5,
        debug_dir: Path | None = None,
    ) -> list[tuple[int, float, float, float, float]]:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        s = self.IMG_SIZE
        resized = cv2.resize(img_rgb, (s, s), interpolation=cv2.INTER_LINEAR)
        tensor  = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor  = tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)                  # (1, 3, S, S)
            probs  = F.softmax(logits, dim=1)[0]         # (3, S, S)

        results: list[tuple[int, float, float, float, float]] = []
        for class_id in range(2):  # 0=drawing, 1=textlabel
            prob_map = probs[class_id + 1].cpu().numpy()  # channel 1/2
            # Resize probability map back to original resolution
            prob_full = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                class_name = self.CLASS_NAMES[class_id]
                # Scale [0,1] → [0,255] and apply colormap so low=dark-blue, high=yellow
                gray = (prob_full * 255).clip(0, 255).astype(np.uint8)
                colored = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
                out = debug_dir / f"{image_path.stem}_{class_name}_prob.png"
                cv2.imwrite(str(out), colored)

            binary = (prob_full >= threshold).astype(np.uint8)
            for x1n, y1n, x2n, y2n in mask_to_boxes(binary, class_id):
                results.append((class_id, x1n, y1n, x2n, y2n))
        return results

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"type": "segmentation", "model_state": self.model.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SegmentationDetector":
        detector = cls()
        data = torch.load(str(path), map_location=detector.device, weights_only=True)
        detector.model.load_state_dict(data["model_state"])
        detector.model.eval()
        return detector
