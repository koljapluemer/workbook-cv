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
   - **Clean Up Image**: Run perspective correction, deskew, denoise, contrast enhancement, and white balance (each step toggleable)
   - **Bounding Box Detection**: Detect text regions with adjustable sensitivity
   - **Draw Annotations**: Manually draw bounding boxes on images
4. All processed versions are saved and accessible via the version dropdown

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
