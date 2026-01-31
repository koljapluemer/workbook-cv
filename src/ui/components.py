import streamlit as st
from pathlib import Path

from ..storage import ImageVersion


def render_navigation(
    image_files: list[Path], current_index: int, key_prefix: str = ""
) -> tuple[int, bool]:
    """
    Render navigation controls (Prev/Next buttons, image counter).

    Returns:
        Tuple of (new_index, changed) where changed indicates if navigation occurred
    """
    col1, col2, col3 = st.columns([1, 1, 3])

    new_index = current_index
    changed = False

    with col1:
        if st.button(
            "← Previous",
            disabled=current_index == 0,
            key=f"{key_prefix}_nav_prev",
        ):
            new_index = current_index - 1
            changed = True

    with col2:
        if st.button(
            "Next →",
            disabled=current_index >= len(image_files) - 1,
            key=f"{key_prefix}_nav_next",
        ):
            new_index = current_index + 1
            changed = True

    with col3:
        st.write(
            f"Image {current_index + 1} of {len(image_files)}: "
            f"**{image_files[current_index].name}**"
        )

    return new_index, changed


def render_version_dropdown(versions: list[ImageVersion]) -> str:
    """
    Render version selection dropdown.

    Args:
        versions: List of available versions (excluding original)

    Returns:
        Selected version ID ("original" or "v001", "v002", etc.)
    """
    # Build options list
    options = ["original"]
    display_names = {"original": "Original"}

    for v in versions:
        options.append(v.id)
        display_names[v.id] = f"{v.display_name} ({v.id})"

    # Use session state to track selection
    if "selected_version" not in st.session_state:
        st.session_state.selected_version = "original"

    # Ensure current selection is valid
    if st.session_state.selected_version not in options:
        st.session_state.selected_version = "original"

    # Find the index of current selection
    current_index = options.index(st.session_state.selected_version)

    selected = st.selectbox(
        "Version",
        options=options,
        index=current_index,
        format_func=lambda x: display_names[x],
    )

    # Update session state
    st.session_state.selected_version = selected

    return selected
