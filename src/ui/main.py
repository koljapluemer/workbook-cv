import streamlit as st
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from ..storage import VersionManager
from ..processing import CleanupPipeline, detect_bounding_boxes
from ..export.yolo_dataset import build_yolo_dataset
from ..export.yolo_auto import train_yolo_model, auto_annotate_test_images
from .components import render_navigation, render_version_dropdown
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
    version_manager: VersionManager,
) -> None:
    image_files = get_image_files(dataset.img_dir)

    if not image_files:
        st.error(f"No images found in {dataset.img_dir}")
        return

    if dataset.read_only:
        st.info("TEST MODE: read-only (annotations and processing disabled)")

    # Initialize session state (per dataset)
    index_key = f"current_index_{dataset.id}"
    version_key = f"selected_version_{dataset.id}"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0
    if version_key not in st.session_state:
        st.session_state[version_key] = "original"

    # Navigation
    new_index, nav_changed = render_navigation(
        image_files,
        st.session_state[index_key],
        key_prefix=f"{dataset.id}",
    )

    if nav_changed:
        st.session_state[index_key] = new_index
        st.session_state[version_key] = "original"
        st.rerun()

    # Current image
    current_file = image_files[st.session_state[index_key]]

    # Get versions for this image
    versions = version_manager.list_versions(current_file)

    # Version dropdown and action buttons in sidebar (main dataset only)
    if not dataset.read_only:
        with st.sidebar:
            st.header("Version Control")
            selected_version = render_version_dropdown(versions)

            st.divider()

            # Clean Up button
            st.subheader("Clean Up Image")

            # Pipeline step checkboxes
            do_perspective = st.checkbox("Perspective Correction", value=True)
            do_deskew = st.checkbox("Deskew", value=True)
            do_denoise = st.checkbox("Denoise", value=True)
            do_contrast = st.checkbox("Contrast Enhancement", value=True)
            do_white_balance = st.checkbox("White Balance", value=False)

            if st.button("Run Cleanup Pipeline"):
                steps = ["Loading image", "Running cleanup pipeline", "Saving version"]
                with progress_operation(steps) as update:
                    update(0)
                    # Load source image
                    if selected_version == "original":
                        source_image = version_manager.load_original(current_file)
                        source_id = "original"
                    else:
                        source = version_manager.get_version(current_file, selected_version)
                        source_image = source.image
                        source_id = selected_version

                    update(1)
                    # Run cleanup pipeline
                    pipeline = CleanupPipeline(
                        do_perspective=do_perspective,
                        do_deskew=do_deskew,
                        do_denoise=do_denoise,
                        do_contrast=do_contrast,
                        do_white_balance=do_white_balance,
                    )
                    cleaned_image, metadata = pipeline.run(source_image)

                    update(2)
                    # Save as new version
                    version = version_manager.create_version(
                        original=current_file,
                        type="cleanup",
                        source=source_id,
                        image=cleaned_image,
                        params=pipeline.get_params(),
                        data=metadata,
                    )

                st.session_state[version_key] = version.id
                st.success(f"Created {version.display_name}")
                st.rerun()

            st.divider()

            # Bounding Box Detection
            st.subheader("Bounding Box Detection")

            # Check if current version is already a bbox result
            is_bbox_version = False
            bbox_sensitivity = None
            if selected_version != "original":
                current_version_meta = next(
                    (v for v in versions if v.id == selected_version), None
                )
                if current_version_meta and current_version_meta.type == "bbox":
                    is_bbox_version = True
                    bbox_sensitivity = current_version_meta.params.get("sensitivity", 5)

            if is_bbox_version:
                st.slider(
                    "Sensitivity",
                    min_value=1,
                    max_value=10,
                    value=bbox_sensitivity,
                    disabled=True,
                )
                st.button("Run Detection", disabled=True)
                st.caption("Already showing bounding box detection result")
            else:
                sensitivity = st.slider("Sensitivity", min_value=1, max_value=10, value=5)

                if st.button("Run Detection"):
                    steps = ["Loading image", "Detecting bounding boxes", "Saving version"]
                    with progress_operation(steps) as update:
                        update(0)
                        # Load source image
                        if selected_version == "original":
                            source_image = version_manager.load_original(current_file)
                            source_id = "original"
                        else:
                            source = version_manager.get_version(current_file, selected_version)
                            source_image = source.image
                            source_id = selected_version

                        update(1)
                        # Run detection
                        result_image, boxes = detect_bounding_boxes(source_image, sensitivity)

                        update(2)
                        # Save as new version
                        version = version_manager.create_version(
                            original=current_file,
                            type="bbox",
                            source=source_id,
                            image=result_image,
                            params={"sensitivity": sensitivity},
                            data={"boxes": boxes},
                        )

                    st.session_state[version_key] = version.id
                    st.success(f"Created {version.display_name} ({len(boxes)} boxes)")
                    st.rerun()

            # Draw Annotations section (available for any version)
            st.divider()
            category = render_category_toggle()

            # Get source image for thumbnails
            if selected_version == "original":
                thumb_source = version_manager.load_original(current_file)
            else:
                thumb_source = version_manager.get_version(current_file, selected_version).image

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
        selected_version = st.session_state[version_key]
        category = "figure"

    # Load and display selected version
    if selected_version == "original":
        display_image = version_manager.load_original(current_file)
        if dataset.read_only:
            st.info("Showing test image with predicted annotations")
        else:
            st.info("Showing original image - Draw rectangles below to annotate")
    else:
        version = version_manager.get_version(current_file, selected_version)
        display_image = version.image

        # Show version info
        info_text = f"Showing: **{version.display_name}** (source: {version.source})"
        if version.type == "bbox":
            box_count = len(version.data.get("boxes", []))
            info_text += f" - {box_count} boxes detected"
        if dataset.read_only:
            info_text += " - Predicted annotations"
        else:
            info_text += " - Draw rectangles below to annotate"
        st.info(info_text)

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

    # Initialize version manager and annotation storage
    version_manager = VersionManager(processed_dir)
    init_annotation_storage(processed_dir)

    tabs = st.tabs([ds.label for ds in datasets])
    for tab, ds in zip(tabs, datasets, strict=True):
        with tab:
            _render_dataset_view(ds, version_manager)
