import streamlit as st
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from ..storage import VersionManager
from ..processing import CleanupPipeline, detect_bounding_boxes
from ..export import export_coco, export_yolo
from .components import render_navigation, render_version_dropdown
from .annotation import render_drawing_canvas, extract_boxes_from_canvas, render_annotated_image


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

        # Draw Annotations section (available for any version)
        st.divider()
        st.subheader("Draw Annotations")
        st.caption("Draw rectangles on the image in the main area, then save.")

        if st.button("Save Annotations", use_container_width=True):
            # Get canvas data from session state
            canvas_data = st.session_state.get("canvas_data")
            if canvas_data:
                scale = canvas_data.get("scale", 1.0)
                boxes = extract_boxes_from_canvas(canvas_data, scale)

                if boxes:
                    # Get source image
                    if selected_version == "original":
                        source_image = version_manager.load_original(current_file)
                    else:
                        source = version_manager.get_version(current_file, selected_version)
                        source_image = source.image

                    # Create annotated image with boxes drawn
                    annotated_image = render_annotated_image(source_image, boxes)

                    # Save annotation version
                    annotation_version = version_manager.create_version(
                        original=current_file,
                        type="annotation",
                        source=selected_version,
                        image=annotated_image,
                        params={"source_version": selected_version},
                        data={
                            "boxes": boxes,
                            "selected_boxes": boxes,
                            "category": "unit",
                        },
                    )

                    st.session_state.selected_version = annotation_version.id
                    st.success(f"Created {annotation_version.display_name} ({len(boxes)} boxes)")
                    st.rerun()
                else:
                    st.warning("No rectangles drawn. Draw some rectangles first.")
            else:
                st.warning("No canvas data available.")

        # Export Annotations section
        st.divider()
        st.subheader("Export Annotations")

        # Count annotated images
        annotated_images = []
        for img_file in image_files:
            img_versions = version_manager.list_versions(img_file)
            for v in img_versions:
                if v.type == "annotation":
                    annotated_images.append((img_file, v.id))
                    break  # Only count first annotation per image

        st.caption(f"{len(annotated_images)} images with annotations")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export COCO", use_container_width=True, disabled=len(annotated_images) == 0):
                _export_coco_annotations(version_manager, annotated_images, processed_dir)
        with col2:
            if st.button("Export YOLO", use_container_width=True, disabled=len(annotated_images) == 0):
                _export_yolo_annotations(version_manager, annotated_images, processed_dir)

    # Load and display selected version
    if selected_version == "original":
        display_image = version_manager.load_original(current_file)
        st.info("Showing original image - Draw rectangles below to annotate")
        existing_boxes = None
    else:
        version = version_manager.get_version(current_file, selected_version)
        display_image = version.image

        # Show version info
        info_text = f"Showing: **{version.display_name}** (source: {version.source})"
        if version.type == "bbox":
            box_count = len(version.data.get("boxes", []))
            info_text += f" - {box_count} boxes detected"
        elif version.type == "annotation":
            selected_count = len(version.data.get("selected_boxes", []))
            info_text += f" - {selected_count} boxes annotated"
        info_text += " - Draw rectangles below to annotate"
        st.info(info_text)

        # Get existing boxes from annotation versions to display
        existing_boxes = None
        if version.type == "annotation":
            existing_boxes = version.data.get("selected_boxes", [])

    # Render drawable canvas instead of static image
    canvas_key = f"canvas_{current_file.stem}_{selected_version}"
    canvas_data = render_drawing_canvas(display_image, existing_boxes, canvas_key)
    st.session_state["canvas_data"] = canvas_data

    # Show box count from canvas
    if canvas_data:
        scale = canvas_data.get("scale", 1.0)
        drawn_boxes = extract_boxes_from_canvas(canvas_data, scale)
        if drawn_boxes:
            st.caption(f"{len(drawn_boxes)} rectangle(s) drawn")


def _export_coco_annotations(
    version_manager: VersionManager,
    annotated_images: list[tuple[Path, str]],
    processed_dir: Path,
) -> None:
    """Export all annotations in COCO JSON format."""
    images_data = []

    for img_file, version_id in annotated_images:
        version = version_manager.get_version(img_file, version_id)
        selected_boxes = version.data.get("selected_boxes", [])

        if not selected_boxes:
            continue

        img_height, img_width = version.image.shape[:2]

        annotations = []
        for box in selected_boxes:
            # COCO format: [x, y, width, height]
            annotations.append({
                "bbox": [box["x"], box["y"], box["w"], box["h"]],
                "category_id": 1,  # "unit" category
            })

        images_data.append({
            "file_name": img_file.name,
            "width": img_width,
            "height": img_height,
            "annotations": annotations,
        })

    categories = [{"id": 1, "name": "unit"}]
    output_path = processed_dir / "exports" / "annotations.json"

    export_coco(images_data, categories, output_path)
    st.success(f"Exported COCO annotations to {output_path}")


def _export_yolo_annotations(
    version_manager: VersionManager,
    annotated_images: list[tuple[Path, str]],
    processed_dir: Path,
) -> None:
    """Export all annotations in YOLO format."""
    images_data = []

    for img_file, version_id in annotated_images:
        version = version_manager.get_version(img_file, version_id)
        selected_boxes = version.data.get("selected_boxes", [])

        if not selected_boxes:
            continue

        img_height, img_width = version.image.shape[:2]

        annotations = []
        for box in selected_boxes:
            annotations.append({
                "bbox": [box["x"], box["y"], box["w"], box["h"]],
                "category_id": 1,  # "unit" category
            })

        images_data.append({
            "file_name": img_file.name,
            "width": img_width,
            "height": img_height,
            "annotations": annotations,
        })

    categories = [{"id": 1, "name": "unit"}]
    output_dir = processed_dir / "exports" / "yolo"

    export_yolo(images_data, categories, output_dir)
    st.success(f"Exported YOLO annotations to {output_dir}")
