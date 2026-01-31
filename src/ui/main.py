import streamlit as st
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from PIL import Image
from ..processing import detect_bounding_boxes
from ..export.yolo_dataset import build_yolo_dataset
from ..export.yolo_auto import train_yolo_model, auto_annotate_test_images
from .components import render_navigation
from .datasets import load_datasets, get_image_files
from .annotation import (
    init_annotation_storage,
    render_drawing_canvas,
    render_category_toggle,
    render_annotation_list,
)


@contextmanager
def progress_operation(steps: list[str]) -> Generator[callable, None, None]:
    """
    Context manager for showing progress through multi-step operations.

    Usage:
        with progress_operation(["Loading", "Processing", "Saving"]) as update:
            update(0)  # Shows "Loading..."
            do_loading()
            update(1)  # Shows "Processing..."
            do_processing()
            update(2)  # Shows "Saving..."
            do_saving()
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(steps)

    def update(step_index: int):
        progress = (step_index + 1) / total
        progress_bar.progress(progress)
        if step_index < total:
            status_text.text(f"{steps[step_index]}...")

    try:
        yield update
    finally:
        progress_bar.progress(1.0)
        status_text.empty()


def _render_dataset_view(
    dataset,
) -> None:
    image_files = get_image_files(dataset.img_dir)

    if not image_files:
        st.error(f"No images found in {dataset.img_dir}")
        return

    if dataset.read_only:
        st.info("TEST MODE: read-only (annotations and processing disabled)")

    # Initialize session state (per dataset)
    index_key = f"current_index_{dataset.id}"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0

    # Navigation
    new_index, nav_changed = render_navigation(
        image_files,
        st.session_state[index_key],
        key_prefix=f"{dataset.id}",
    )

    if nav_changed:
        st.session_state[index_key] = new_index
        st.rerun()

    # Current image
    current_file = image_files[st.session_state[index_key]]

    # Sidebar controls (main dataset only)
    if not dataset.read_only:
        with st.sidebar:
            st.header("Annotations")

            st.subheader("Auto-detected boxes")
            auto_boxes_key = f"auto_boxes_{dataset.id}"
            auto_sensitivity_key = f"auto_sensitivity_{dataset.id}"
            show_auto_boxes = st.toggle(
                "Show auto-detected boxes",
                value=st.session_state.get(auto_boxes_key, False),
                key=auto_boxes_key,
            )
            sensitivity = st.slider(
                "Sensitivity",
                min_value=1,
                max_value=10,
                value=st.session_state.get(auto_sensitivity_key, 5),
                key=auto_sensitivity_key,
                disabled=not show_auto_boxes,
            )

            st.divider()
            category = render_category_toggle()

            # Get source image for thumbnails
            thumb_source = _load_original_image(current_file)

            # Render annotation list with thumbnails
            deleted = render_annotation_list(current_file.stem, thumb_source, read_only=False)
            if deleted:
                st.rerun()

            st.divider()
            st.caption("Annotations auto-saved to processed/annotations/")

            st.divider()
            st.subheader("YOLO")
            base_dir = Path(__file__).parent.parent.parent
            output_dir = base_dir / "datasets" / "wordbook"

            if st.button("Run Auto-Annotation Pipeline"):
                try:
                    steps = ["Preparing dataset", "Training model", "Auto-annotating test data"]
                    with progress_operation(steps) as update:
                        update(0)
                        result = build_yolo_dataset(base_dir, output_dir)
                        update(1)
                        weights_path = train_yolo_model(base_dir, output_dir)
                        update(2)
                        auto_result = auto_annotate_test_images(base_dir, weights_path)
                    st.success(
                        f"Done. Train={result.train_count}, val={result.val_count}, "
                        f"test={result.test_count}; wrote {auto_result.annotated_count} annotations."
                    )
                    if result.skipped_annotations:
                        st.caption(
                            f"Skipped {result.skipped_annotations} annotations with no matching image."
                        )
                    if auto_result.skipped_count:
                        st.caption(
                            f"Skipped {auto_result.skipped_count} images with no detections."
                        )
                except Exception as exc:
                    st.error(f"Auto-annotation pipeline failed: {exc}")
    else:
        category = "figure"
        show_auto_boxes = False
        sensitivity = 5

    # Load and display image
    original_image = _load_original_image(current_file)
    if dataset.read_only:
        st.info("Showing test image with predicted annotations")
    else:
        st.info("Showing original image - Draw rectangles below to annotate")

    display_image = original_image
    if not dataset.read_only and show_auto_boxes:
        display_image, auto_boxes = detect_bounding_boxes(original_image, sensitivity)
        st.caption(f"Auto-detected {len(auto_boxes)} boxes")

    # Render drawable canvas (auto-persists annotations)
    canvas_key = f"canvas_{current_file.stem}"
    render_drawing_canvas(
        display_image,
        current_file.stem,
        category,
        canvas_key,
        read_only=dataset.read_only,
    )

    if dataset.read_only:
        st.subheader("Annotations")
        deleted = render_annotation_list(
            current_file.stem, display_image, read_only=True
        )
        if deleted:
            st.rerun()


def run_app():
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Textbook Scanner", layout="wide")

    # Setup directories
    base_dir = Path(__file__).parent.parent.parent
    processed_dir = base_dir / "processed"

    datasets = load_datasets(base_dir)

    if not datasets:
        st.error("No datasets found")
        return

    init_annotation_storage(processed_dir)

    tabs = st.tabs([ds.label for ds in datasets])
    for tab, ds in zip(tabs, datasets, strict=True):
        with tab:
            _render_dataset_view(ds)


def _load_original_image(path: Path) -> np.ndarray:
    """Load an image as RGB numpy array."""
    image = Image.open(path)
    return np.array(image.convert("RGB"))
