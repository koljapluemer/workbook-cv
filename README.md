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
4. All processed versions are saved and accessible via the version dropdown

## Folder Structure

```
wordbook-cv/
├── app.py                  # Entry point
├── img/                    # Source images (user-provided)
├── processed/              # Versioned outputs (auto-generated)
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
        ├── components.py       # Reusable UI components
        └── main.py             # Main app layout
```
