import streamlit as st
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from ..storage import VersionManager
from ..processing import CleanupPipeline, detect_bounding_boxes
from .components import render_navigation, render_version_dropdown


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


def get_image_files(img_dir: Path) -> list[Path]:
    """Get all image files from the directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    files = [f for f in img_dir.iterdir() if f.suffix.lower() in extensions]
    return sorted(files)


def run_app():
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Textbook Scanner", layout="wide")

    # Setup directories
    base_dir = Path(__file__).parent.parent.parent
    img_dir = base_dir / "img"
    processed_dir = base_dir / "processed"

    if not img_dir.exists():
        st.error(f"Image directory not found: {img_dir}")
        return

    image_files = get_image_files(img_dir)

    if not image_files:
        st.error("No images found in img/ directory")
        return

    # Initialize version manager
    version_manager = VersionManager(processed_dir)

    # Initialize session state
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "selected_version" not in st.session_state:
        st.session_state.selected_version = "original"

    # Navigation
    new_index, nav_changed = render_navigation(
        image_files, st.session_state.current_index
    )

    if nav_changed:
        st.session_state.current_index = new_index
        st.session_state.selected_version = "original"
        st.rerun()

    # Current image
    current_file = image_files[st.session_state.current_index]

    # Get versions for this image
    versions = version_manager.list_versions(current_file)

    # Version dropdown and action buttons in sidebar
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

            st.session_state.selected_version = version.id
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

                st.session_state.selected_version = version.id
                st.success(f"Created {version.display_name} ({len(boxes)} boxes)")
                st.rerun()

    # Load and display selected version
    if selected_version == "original":
        display_image = version_manager.load_original(current_file)
        st.info("Showing original image")
    else:
        version = version_manager.get_version(current_file, selected_version)
        display_image = version.image

        # Show version info
        info_text = f"Showing: **{version.display_name}** (source: {version.source})"
        if version.type == "bbox":
            box_count = len(version.data.get("boxes", []))
            info_text += f" - {box_count} boxes detected"
        st.info(info_text)

    st.image(display_image, use_container_width=True)
