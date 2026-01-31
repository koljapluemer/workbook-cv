import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def get_image_files(img_dir: Path) -> list[Path]:
    """Get all image files from the directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    files = [f for f in img_dir.iterdir() if f.suffix.lower() in extensions]
    return sorted(files)


def detect_bounding_boxes(image: np.ndarray) -> np.ndarray:
    """
    Detect text blocks and graphics in a textbook scan using contour detection.
    Returns the image with red bounding boxes drawn around detected elements.
    """
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding for better results on scanned documents
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate to connect nearby text/elements into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(binary, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw bounding boxes
    img_height, img_width = image.shape[:2]
    min_area = (img_width * img_height) * 0.001  # Min 0.1% of image area
    max_area = (img_width * img_height) * 0.95  # Max 95% of image area

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Filter by area and aspect ratio
        if min_area < area < max_area:
            # Draw red bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return result


def main():
    st.set_page_config(page_title="Textbook Scanner", layout="wide")
    st.title("Textbook Image Browser")

    img_dir = Path(__file__).parent / "img"

    if not img_dir.exists():
        st.error(f"Image directory not found: {img_dir}")
        return

    image_files = get_image_files(img_dir)

    if not image_files:
        st.error("No images found in img/ directory")
        return

    # Initialize session state
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "show_boxes" not in st.session_state:
        st.session_state.show_boxes = False

    # Navigation
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    with col1:
        if st.button("← Previous", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.session_state.show_boxes = False
            st.rerun()

    with col2:
        if st.button(
            "Next →", disabled=st.session_state.current_index >= len(image_files) - 1
        ):
            st.session_state.current_index += 1
            st.session_state.show_boxes = False
            st.rerun()

    with col3:
        st.write(
            f"Image {st.session_state.current_index + 1} of {len(image_files)}: "
            f"**{image_files[st.session_state.current_index].name}**"
        )

    with col4:
        if st.button("Run Bounding Box Detection"):
            st.session_state.show_boxes = True
            st.rerun()

    # Load and display image
    current_file = image_files[st.session_state.current_index]
    image = Image.open(current_file)
    image_array = np.array(image.convert("RGB"))

    if st.session_state.show_boxes:
        image_array = detect_bounding_boxes(image_array)
        st.info("Showing detected bounding boxes (red rectangles)")

    st.image(image_array, use_container_width=True)


if __name__ == "__main__":
    main()
