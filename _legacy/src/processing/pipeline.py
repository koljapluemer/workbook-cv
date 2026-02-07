import numpy as np

from .deskew import deskew
from .perspective import perspective_correction
from .enhance import denoise, enhance_contrast, white_balance


class CleanupPipeline:
    """
    Orchestrates the cleanup pipeline stages.

    Stages (in order):
    1. Page Detection & Perspective Correction
    2. Deskew (rotation correction)
    3. Denoise
    4. Contrast Enhancement (CLAHE)
    5. White Balance (optional)
    """

    def __init__(
        self,
        do_perspective: bool = True,
        do_deskew: bool = True,
        do_denoise: bool = True,
        do_contrast: bool = True,
        do_white_balance: bool = False,
        denoise_strength: int = 10,
        clahe_clip_limit: float = 2.0,
    ):
        self.do_perspective = do_perspective
        self.do_deskew = do_deskew
        self.do_denoise = do_denoise
        self.do_contrast = do_contrast
        self.do_white_balance = do_white_balance
        self.denoise_strength = denoise_strength
        self.clahe_clip_limit = clahe_clip_limit

    def run(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Run the cleanup pipeline on an image.

        Args:
            image: RGB numpy array

        Returns:
            Tuple of (cleaned_image, metadata_dict)
        """
        result = image
        metadata = {
            "stages_applied": [],
            "perspective": {},
            "deskew": {},
            "denoise": {},
            "contrast": {},
            "white_balance": {},
        }

        # Stage 1: Perspective correction
        if self.do_perspective:
            result, stage_meta = perspective_correction(result)
            metadata["perspective"] = stage_meta
            if stage_meta.get("page_detected"):
                metadata["stages_applied"].append("perspective")

        # Stage 2: Deskew
        if self.do_deskew:
            result, stage_meta = deskew(result)
            metadata["deskew"] = stage_meta
            if abs(stage_meta.get("angle", 0)) > 0.1:
                metadata["stages_applied"].append("deskew")

        # Stage 3: Denoise
        if self.do_denoise:
            result, stage_meta = denoise(result, strength=self.denoise_strength)
            metadata["denoise"] = stage_meta
            metadata["stages_applied"].append("denoise")

        # Stage 4: Contrast enhancement
        if self.do_contrast:
            result, stage_meta = enhance_contrast(result, clip_limit=self.clahe_clip_limit)
            metadata["contrast"] = stage_meta
            metadata["stages_applied"].append("contrast")

        # Stage 5: White balance (optional)
        if self.do_white_balance:
            result, stage_meta = white_balance(result)
            metadata["white_balance"] = stage_meta
            metadata["stages_applied"].append("white_balance")

        return result, metadata

    def get_params(self) -> dict:
        """Get the current pipeline parameters."""
        return {
            "perspective": self.do_perspective,
            "deskew": self.do_deskew,
            "denoise": self.do_denoise,
            "contrast": self.do_contrast,
            "white_balance": self.do_white_balance,
            "denoise_strength": self.denoise_strength,
            "clahe_clip_limit": self.clahe_clip_limit,
        }
