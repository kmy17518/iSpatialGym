"""The goal image: the TRO's ``goal_observation.rgb_path`` (``run_isg --goal-image``) when the instance has one, else
the configured PNG (``harness.run`` passes the demo's ``final_rgb.png``: the oracle's settled terminal head frame)."""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np


def load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))


def goal_image(tro_path: Optional[str], configured: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """``(rgb, source path)``; ``(None, None)`` when there is no goal image."""
    if tro_path and os.path.exists(tro_path):
        try:
            g = (json.load(open(tro_path)).get("ispatialgym") or {}).get("goal_observation") or {}
        except (OSError, ValueError):
            g = {}
        p = g.get("rgb_path")
        if p:
            p = p if os.path.isabs(p) else os.path.join(os.path.dirname(tro_path), p)
            if os.path.exists(p):
                return load_rgb(p), p
    if configured and os.path.exists(configured):
        return load_rgb(configured), configured
    return None, None
