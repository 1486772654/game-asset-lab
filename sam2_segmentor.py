"""Optional SAM 2.1 image segmenter, loaded only when configured."""
from __future__ import annotations

from pathlib import Path
import numpy as np

class Sam21Segmentor:
    def __init__(self, checkpoint: str, config: str, device: str = "cuda"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))

    def masks_for(self, image_rgb: np.ndarray, boxes: list[list[int]]) -> list[np.ndarray]:
        self.predictor.set_image(image_rgb)
        output = []
        for box in boxes:
            masks, _, _ = self.predictor.predict(box=np.array(box), multimask_output=False)
            output.append(masks[0].astype(bool))
        return output

_INSTANCE = None
_ERROR = None

def get_segmentor(checkpoint: str, config: str, device: str = "cuda"):
    global _INSTANCE, _ERROR
    if _INSTANCE is not None:
        return _INSTANCE
    if not checkpoint or not Path(checkpoint).is_file() or not config:
        return None
    try:
        _INSTANCE = Sam21Segmentor(checkpoint, config, device)
        return _INSTANCE
    except Exception as exc:
        _ERROR = str(exc)
        return None

def last_error():
    return _ERROR
