"""Auto Reframe -- following a subject while changing the aspect ratio.

The crop window is a fixed size derived from the target ratio; only its centre
moves. Two things keep that movement watchable:

    smoothing   a moving average over a window of seconds, because raw
                detection centres jitter frame to frame and a crop that tracks
                them exactly is unwatchable
    holding     when the subject is missing, the window stays where it was
                instead of snapping to a default

This is also the operation that cannot be confirmed from one frame. The subject
can be correct and the result still wrong -- drifting to the edge, swinging
between people -- which is why the confirmation step shows several frames and
the centre trajectory rather than a single highlight.
"""

from __future__ import annotations

from dataclasses import dataclass

from cutpilot.schemas.track import Box, TrackBoxes


@dataclass(frozen=True)
class CropSize:
    """The window's dimensions in normalized units; only the centre moves."""

    w: float
    h: float

    @property
    def moves_horizontally(self) -> bool:
        return self.w < 1.0


def parse_aspect_ratio(value: str) -> float:
    """Parse "9:16" into 0.5625. Also accepts a bare float."""
    if ":" in value:
        left, right = value.split(":", 1)
        w, h = float(left), float(right)
        if w <= 0 or h <= 0:
            raise ValueError(f"bad aspect ratio: {value}")
        return w / h
    ratio = float(value)
    if ratio <= 0:
        raise ValueError(f"bad aspect ratio: {value}")
    return ratio


def crop_size_for(source_w: int, source_h: int, target_ratio: float) -> CropSize:
    """Largest window of the target ratio that fits, in normalized units.

    A wider source than the target loses width; a taller one loses height. One
    axis always stays full, so the window only ever slides along the other.
    """
    source_ratio = source_w / source_h
    if source_ratio > target_ratio:
        return CropSize(w=target_ratio / source_ratio, h=1.0)
    return CropSize(w=1.0, h=source_ratio / target_ratio)


def _centre_at(boxes: TrackBoxes, t: float) -> tuple[float, float] | None:
    box = boxes.at(t)
    return box.center if box else None


def centre_trajectory(
    boxes: TrackBoxes,
    *,
    duration: float,
    fps: float,
) -> list[tuple[float, float] | None]:
    """Subject centre per frame, with None where the subject is not visible."""
    count = max(1, int(round(duration * fps)))
    return [_centre_at(boxes, index / fps) for index in range(count)]


def smooth(
    centres: list[tuple[float, float] | None],
    *,
    fps: float,
    window_seconds: float = 1.0,
) -> list[tuple[float, float]]:
    """Moving average over the trajectory, holding through gaps.

    Gaps are filled from the last known centre (and, at the start, from the
    first one) before averaging, so a brief loss of detection does not pull the
    crop toward a default position and back.
    """
    if not centres:
        return []

    first = next((c for c in centres if c is not None), (0.5, 0.5))
    filled: list[tuple[float, float]] = []
    last = first
    for centre in centres:
        last = centre if centre is not None else last
        filled.append(last)

    radius = max(0, int(round(window_seconds * fps / 2)))
    if radius == 0:
        return filled

    smoothed: list[tuple[float, float]] = []
    for index in range(len(filled)):
        lo = max(0, index - radius)
        hi = min(len(filled), index + radius + 1)
        window = filled[lo:hi]
        smoothed.append(
            (
                sum(c[0] for c in window) / len(window),
                sum(c[1] for c in window) / len(window),
            )
        )
    return smoothed


def window_at(centre: tuple[float, float], size: CropSize) -> Box:
    """Place the window on a centre, kept inside the frame."""
    x = min(max(centre[0] - size.w / 2, 0.0), 1.0 - size.w)
    y = min(max(centre[1] - size.h / 2, 0.0), 1.0 - size.h)
    return Box(x=x, y=y, w=size.w, h=size.h)


def crop_trajectory(
    boxes: TrackBoxes,
    *,
    source_w: int,
    source_h: int,
    target_ratio: float,
    duration: float,
    fps: float,
    window_seconds: float = 1.0,
) -> list[Box]:
    """Per-frame crop windows for one subject. Index is the frame number."""
    size = crop_size_for(source_w, source_h, target_ratio)
    centres = smooth(
        centre_trajectory(boxes, duration=duration, fps=fps),
        fps=fps,
        window_seconds=window_seconds,
    )
    return [window_at(centre, size) for centre in centres]
