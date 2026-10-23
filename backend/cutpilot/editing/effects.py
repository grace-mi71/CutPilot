"""Pixel operations that hide a region.

Region-based rather than pixel-accurate masking, which is the decision recorded
in docs/decisions/0001: it is simpler to get right, and covering slightly more
than the subject is the safer error for privacy work. A mask trimmed exactly to
a face leaves the jaw, ears and hairline in frame.

`strength` runs 1-100 and means the same thing in both effects: higher hides
more. Effect sizes scale with the region, so a distant face and a close-up get
comparable treatment rather than the same kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

MIN_STRENGTH = 1
MAX_STRENGTH = 100

# Below this the region is a handful of pixels; blurring it is pointless and
# cv2 rejects kernels larger than the image anyway.
MIN_REGION_PX = 4


def clamp_strength(strength: int | None, default: int) -> int:
    value = default if strength is None else strength
    return max(MIN_STRENGTH, min(MAX_STRENGTH, value))


def _clip_region(
    frame_w: int, frame_h: int, x: int, y: int, w: int, h: int
) -> tuple[int, int, int, int] | None:
    """Intersect a region with the frame, or None if nothing is left."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + w), min(frame_h, y + h)
    if x1 - x0 < MIN_REGION_PX or y1 - y0 < MIN_REGION_PX:
        return None
    return x0, y0, x1 - x0, y1 - y0


def blur_region(frame: np.ndarray, x: int, y: int, w: int, h: int, strength: int) -> bool:
    """Gaussian-blur a rectangle in place. Returns whether anything was drawn."""
    import cv2

    clipped = _clip_region(frame.shape[1], frame.shape[0], x, y, w, h)
    if clipped is None:
        return False
    x, y, w, h = clipped

    # Kernel scales with the region so the result reads the same whether the
    # subject is near or far. Odd and at least 3, as cv2 requires.
    kernel = int(min(w, h) * strength / 100)
    kernel = max(3, kernel | 1)
    frame[y : y + h, x : x + w] = cv2.GaussianBlur(frame[y : y + h, x : x + w], (kernel, kernel), 0)
    return True


def mosaic_region(frame: np.ndarray, x: int, y: int, w: int, h: int, strength: int) -> bool:
    """Pixelate a rectangle in place. Returns whether anything was drawn."""
    import cv2

    clipped = _clip_region(frame.shape[1], frame.shape[0], x, y, w, h)
    if clipped is None:
        return False
    x, y, w, h = clipped

    # Strength is inverted into a block count: stronger means fewer, larger
    # blocks. 20 -> 5 blocks across the region, which erases a face outright.
    blocks = max(2, int(round(MAX_STRENGTH / max(1, strength))))
    small_w = max(1, min(w, blocks))
    small_h = max(1, min(h, blocks))

    region = frame[y : y + h, x : x + w]
    shrunk = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
    frame[y : y + h, x : x + w] = cv2.resize(shrunk, (w, h), interpolation=cv2.INTER_NEAREST)
    return True


def apply_effect(frame: np.ndarray, op: str, x: int, y: int, w: int, h: int, strength: int) -> bool:
    if op == "BLUR":
        return blur_region(frame, x, y, w, h, strength)
    if op == "MOSAIC":
        return mosaic_region(frame, x, y, w, h, strength)
    raise ValueError(f"not a privacy effect: {op}")
