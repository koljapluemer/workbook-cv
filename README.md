# Wordbook-CV

Textbook image processing and annotation tool with a Streamlit UI.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Running

```bash
# Install dependencies
uv sync

# Run the app
uv run streamlit run app.py
```

Or with pip:

```bash
pip install -e .
streamlit run app.py
```

## Usage

1. Place images in the `img/` folder
2. Run the app and browse images with prev/next navigation
3. Use the sidebar to:
   - **Auto-detected boxes**: Toggle red boxes with adjustable sensitivity
   - **OCR overlay**: Toggle yellow text boxes with adjustable confidence
   - **Draw Annotations**: Manually draw bounding boxes on images
4. Annotations are saved and accessible per image

### Test Tab (Read-only)

If `img/test-data/` exists and contains images, a **Test** tab appears in the UI.

- Test mode is read-only: no cleanup, detection, or annotation writes
- Existing annotations from `processed/annotations/` are rendered as overlays
- Use it to visually inspect model or auto-annotation outputs without modifying data

## Annotation System

Draw rectangles directly on images to create training data for object detection (YOLO, etc.).

### Categories

| Category | Color | YOLO Class ID |
|----------|-------|---------------|
| `figure` | Green | 0 |
| `figure_label` | Blue | 1 |

### Drawing Annotations

1. Select a category using the radio toggle in the sidebar
2. Draw rectangles on the canvas in the main area
3. Annotations auto-save immediately (no save button needed)
4. Delete annotations using the X button in the sidebar list

### Output Formats

Annotations are saved to `processed/annotations/` in two formats simultaneously:

#### COCO JSON (source of truth)

Per-image JSON file used for loading annotations back into the app.

```
processed/annotations/{image_name}.json
```

```json
{
  "image": {
    "file_name": "page001.jpg",
    "width": 1920,
    "height": 1080
  },
  "annotations": [
    {
      "id": 0,
      "bbox": [100, 200, 150, 80],
      "category_id": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "figure"},
    {"id": 1, "name": "figure_label"}
  ]
}
```

- `bbox`: `[x, y, width, height]` in absolute pixels

#### YOLO TXT

Per-image text file for direct use with YOLO training.

```
processed/annotations/{image_name}.txt
```

```
0 0.450000 0.320000 0.120000 0.080000
1 0.710000 0.550000 0.150000 0.100000
```

Format: `class_id center_x center_y width height`
- All coordinates normalized to 0-1 range
- One line per bounding box

## Folder Structure

```
wordbook-cv/
├── app.py                  # Entry point
├── img/                    # Source images (user-provided)
├── processed/              # Auto-generated outputs
│   ├── {image}/            # Versioned processed images
│   └── annotations/        # Annotation files
│       ├── {image}.json    # COCO format (source of truth)
│       └── {image}.txt     # YOLO format
└── src/
    ├── processing/         # Image processing algorithms
    │   ├── bounding_boxes.py   # Text region detection
    │   ├── deskew.py           # Rotation correction
    │   ├── enhance.py          # Denoise, contrast, white balance
    │   ├── perspective.py      # Page detection & perspective fix
    │   └── pipeline.py         # CleanupPipeline orchestrator
    ├── storage/            # Version management
    │   ├── manager.py          # VersionManager class
    │   └── version.py          # Version data model
    └── ui/                 # Streamlit interface
        ├── annotation.py       # Canvas drawing & persistence
        ├── components.py       # Reusable UI components
        └── main.py             # Main app layout
```

## Scripts

### Diversity Heatmap (`scripts/diversity_heatmap.py`)

Slides a square kernel over each train image and produces an **RGB heatmap** where each
channel encodes a different signal:

| Channel | Signal | Method |
|---------|--------|--------|
| **R** | Color diversity | Mean per-channel std dev in the sliding kernel (O(n) box filter) |
| **G** | Text likelihood | OCR run with 5 Tesseract PSM modes; per-pixel sum of `confidence/100` across all boxes covering that pixel |
| **B** | HSV color closeness | Euclidean distance in normalised HSV space to a target color (default H=220°, S=65%, V=46%); H distance is circular |

Pure red → visually rich area with no text. Pure green → clear text. Pure blue → pixels near hue 220°.

Outputs per source image:

| Folder | Contents |
|--------|----------|
| `heatmaps/` | Full RGB heatmap (R=diversity, G=text, B=color closeness) |
| `combined/` | Original scan (left) + full heatmap (right) |
| `channel_r/` | R channel only — color diversity, greyscale |
| `channel_g/` | G channel only — text likelihood, greyscale |
| `channel_b/` | B channel only — HSV color closeness, greyscale |

All outputs land under the `--output-dir` root (default `data/heatmaps/`).

```bash
# Default: kernel size 32, reads from src/img/train/, writes to data/heatmaps/
uv run python scripts/diversity_heatmap.py

# Custom kernel size
uv run python scripts/diversity_heatmap.py --kernel-size 64

# Skip OCR (G=0, fast) — diversity only
uv run python scripts/diversity_heatmap.py --skip-ocr

# Custom input/output directories
uv run python scripts/diversity_heatmap.py --input-dir src/img/validate --output-dir data/validate_heatmaps
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel-size N` | `32` | Kernel side length in pixels |
| `--input-dir DIR` | `src/img/train` | Directory of input images |
| `--output-dir DIR` | `data/heatmaps` | Output root directory |
| `--skip-ocr` | off | Skip OCR (G channel = 0); faster for diversity-only runs |
| `--target-hue DEG` | `220` | Target hue in degrees 0–360 for the B channel |
| `--target-sat S` | `0.65` | Target saturation 0–1 for the B channel |
| `--target-val V` | `0.46` | Target brightness/value 0–1 for the B channel |

---

## Model Training (YOLO)

This project already writes YOLO labels to `processed/annotations/{image}.txt` as you annotate.
To train a YOLO model and auto-annotate test data, use the built-in UI buttons (no manual copying).

### Dependencies

Ultralytics is included in the project dependencies. Run `uv sync` (or `pip install -e .`)
and you are ready to train and auto-annotate from the UI.

### Prepare a dataset layout (built-in)

In the Streamlit sidebar, click **Prepare YOLO Dataset**. This creates:

```
datasets/wordbook/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

It uses annotated images from `img/` (train/val split) and, if present, `img/test-data`
as the test set. Labels are pulled directly from `processed/annotations/`.

### Train + Auto-annotate Test Data

Use the sidebar buttons:

1. **Prepare YOLO Dataset**
2. **Train YOLO Model**
3. **Auto-annotate Test Data**

The auto-annotation step writes COCO JSON + YOLO TXT into `processed/annotations/`
for every image in `img/test-data`. Switch to the **Test** tab to review the overlays.

### Review Results

After **Auto-annotate Test Data**, switch to the **Test** tab to see predicted
boxes drawn on the test images.
