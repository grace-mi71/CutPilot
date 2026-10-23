"""Executor -- applies an approved EditPlan: reframe, blur/mosaic, render.

This is the half of the system the agent never touches. The plan arrives as
data, already grounded and already approved, and nothing here asks a model
anything: the same plan over the same video produces the same file every time.
That is the point of keeping planning and execution apart.

Order within a frame matters. Privacy effects are applied in source
coordinates, then the reframe crop is taken. Doing it the other way would mean
transforming every box into crop space first, and a subject that leaves the
crop would silently stop being covered partway through.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cutpilot.editing import effects, reframe
from cutpilot.editing.encoder import FrameWriter, has_audio
from cutpilot.schemas.plan import EditPlan, Operation
from cutpilot.schemas.track import Box, Track, TrackBoxes
from cutpilot.schemas.video import VideoAsset
from cutpilot.vision.resolve import resolve_region_box, resolve_selector


@dataclass
class RenderResult:
    target: Path
    frames: int = 0
    output_width: int = 0
    output_height: int = 0
    regions_drawn: int = 0
    regions_cropped_away: int = 0
    """Effects that landed outside the reframe window -- work the output never
    shows. A non-zero count here is the `out_of_frame` warning the user should
    have seen before approving."""
    per_operation: dict[int, int] = field(default_factory=dict)

    @property
    def aspect(self) -> str:
        return f"{self.output_width}x{self.output_height}"


def _even(value: int) -> int:
    """H.264 refuses odd dimensions."""
    return value if value % 2 == 0 else value - 1


def _crop_origin(
    window: Box, frame_w: int, frame_h: int, out_w: int, out_h: int
) -> tuple[int, int]:
    """Pixel origin for a crop of exactly (out_w, out_h), kept inside the frame."""
    x = int(round(window.x * frame_w))
    y = int(round(window.y * frame_h))
    return max(0, min(x, frame_w - out_w)), max(0, min(y, frame_h - out_h))


def execute(
    plan: EditPlan,
    asset: VideoAsset,
    tracks: list[Track],
    boxes: dict[str, TrackBoxes],
    target: Path,
    *,
    source: Path | None = None,
    blur_strength: int = 25,
    mosaic_strength: int = 20,
    smoothing_window: float = 1.0,
    keep_audio: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> RenderResult:
    """Render `plan` over `source` (the original unless told otherwise).

    Passing the proxy as `source` produces a fast low-resolution preview from
    exactly the same code path as the final render, which is what makes the
    preview trustworthy.
    """
    import cv2

    src = source or Path(asset.source_path)
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {src}")

    frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = asset.fps

    # -- reframe geometry, computed once -----------------------------------
    reframe_op = plan.reframe
    windows: list[Box] = []
    out_w, out_h = frame_w, frame_h

    if reframe_op is not None:
        subject_id = reframe_op.selector.include[0]
        targets = resolve_selector(reframe_op.selector, tracks)
        if not targets:
            raise ValueError(f"reframe subject {subject_id} resolved to nothing")
        subject_boxes = boxes.get(targets[0].id)
        if subject_boxes is None:
            raise ValueError(f"no boxes for reframe subject {targets[0].id}")

        ratio = reframe.parse_aspect_ratio(reframe_op.aspect_ratio or "9:16")
        size = reframe.crop_size_for(frame_w, frame_h, ratio)
        out_w = _even(int(round(size.w * frame_w)))
        out_h = _even(int(round(size.h * frame_h)))
        windows = reframe.crop_trajectory(
            subject_boxes,
            source_w=frame_w,
            source_h=frame_h,
            target_ratio=ratio,
            duration=asset.duration,
            fps=fps,
            window_seconds=smoothing_window,
        )

    # -- resolve each privacy op once, not per frame -----------------------
    privacy: list[tuple[int, Operation, list[Track], int]] = []
    for index, op in enumerate(plan.operations):
        if not op.is_privacy_op:
            continue
        targets = resolve_selector(op.selector, tracks)
        strength = effects.clamp_strength(
            op.strength, blur_strength if op.op == "BLUR" else mosaic_strength
        )
        privacy.append((index, op, targets, strength))

    result = RenderResult(target=target, output_width=out_w, output_height=out_h)
    audio = src if (keep_audio and has_audio(src)) else None

    with FrameWriter(target, width=out_w, height=out_h, fps=fps, audio_from=audio) as writer:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = index / fps

            window = windows[min(index, len(windows) - 1)] if windows else None

            for op_index, op, targets, strength in privacy:
                start, end = op.time_range.resolve(asset.duration)
                if not (start <= t < end):
                    continue
                for track in targets:
                    if not track.present_at(t):
                        continue
                    region = resolve_region_box(track, op.region, t, tracks, boxes)
                    if region is None:
                        continue

                    if window is not None and window.intersection_area(region) == 0.0:
                        result.regions_cropped_away += 1

                    x, y, w, h = region.clamped().scaled_to(frame_w, frame_h)
                    if effects.apply_effect(frame, op.op, x, y, w, h, strength):
                        result.regions_drawn += 1
                        result.per_operation[op_index] = result.per_operation.get(op_index, 0) + 1

            if window is not None:
                ox, oy = _crop_origin(window, frame_w, frame_h, out_w, out_h)
                frame = frame[oy : oy + out_h, ox : ox + out_w]

            writer.write(frame)
            index += 1
            if progress is not None and total:
                progress(index, total)

    capture.release()
    result.frames = index
    return result
