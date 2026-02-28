# Wordbook-CV

Textbook image processing and annotation tool with a Streamlit UI.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Tesseract OCR installed system-wide (`sudo apt install tesseract-ocr` on Ubuntu)

## Running

```bash
# Install dependencies
uv sync

# Run the app
uv run streamlit run app.py
```

## Usage

The app has three tabs:

### Labeling Tab

1. Place training images in `src/img/train/`
2. Select an image and adjust **Detection Sensitivity** and merge gap sliders
3. The app auto-detects feature regions and runs OCR
4. Label each detected region as `label`, `figure`, or `irrelevant`
5. Click **Save Labels** — stored in `data/labels.json`

### Train & Validate Tab

1. Place validation images in `src/img/validate/`
2. Click **Train Network and Check** — trains a RandomForest on your saved labels
3. Predictions (label + confidence) appear for every detected region across all validation images
4. Results persist to disk in `data/validation/` and reload on next app start

### Heatmaps Tab

Generates RGB-encoded heatmaps from source images and lets you draw YOLO bounding boxes
on them for detector training.

#### Channel encoding

| Channel | Signal | Method |
|---------|--------|--------|
| **R** | Color diversity | Mean per-channel std dev in a sliding kernel (O(n) box filter) |
| **G** | Text likelihood | OCR with 5 Tesseract PSM modes × raw + binarized image; per-pixel sum of `confidence/100` |
| **B** | HSV color closeness | Euclidean distance in normalised HSV space to a target color (default H=220°, S=65%, V=46%) |

Pure red → visually rich, no text. Pure green → clear text. Pure blue → pixels near the target hue.

#### Generate

1. Adjust settings in **Generation Settings** (kernel size, skip OCR, target HSV)
2. Click **Generate Heatmaps** — processes all images in `src/img/train/` and `src/img/validate/`
3. Progress is shown inline; outputs land in `data/heatmaps/`

#### Label

1. Select a heatmap from the dropdown (split: train/val is shown automatically)
2. Choose a draw class: `drawing` (red) or `textlabel` (blue)
3. Draw rectangles on the canvas — boxes auto-save as YOLO `.txt` files in `data/heatmap_labels/`
4. **Clear all boxes** removes all annotations for the current image

## Folder Structure

```
wordbook-cv/
├── app.py                      # Entry point
├── src/
│   ├── config.py               # Path constants and defaults
│   ├── heatmap/
│   │   └── generation.py       # Diversity, OCR likelihood, HSV closeness
│   ├── classifier/             # RandomForest training & prediction
│   ├── analysis/               # OCR, histogram, coverage
│   ├── extraction/             # Feature rectangle detection
│   ├── labeling/               # Label storage (JSON)
│   └── ui/
│       └── app.py              # Streamlit UI (all three tabs)
├── src/img/
│   ├── train/                  # Training images (user-provided)
│   └── validate/               # Validation images (user-provided)
└── data/
    ├── heatmaps/
    │   ├── rgb/                # Full RGB heatmaps
    │   ├── combined/           # Original scan + heatmap side-by-side
    │   ├── channel_r/          # Color diversity (grayscale)
    │   ├── channel_g/          # Text likelihood (grayscale)
    │   └── channel_b/          # HSV closeness (grayscale)
    ├── heatmap_labels/         # YOLO labels for heatmap images
    ├── validation/             # Validation results & feature images
    ├── models/                 # Trained models
    ├── labels.json             # Feature labels (source of truth)
    └── settings.json           # Persisted UI settings
```

## Annotation System

Feature labels are stored in `data/labels.json` and used to train the RandomForest classifier.

### YOLO Heatmap Labels

Bounding boxes drawn in the **Heatmaps** tab are saved to `data/heatmap_labels/{stem}.txt`
in standard YOLO format:

```
class_id center_x center_y width height
```

- All coordinates normalised to 0–1
- Class 0 = `drawing`, Class 1 = `textlabel`
