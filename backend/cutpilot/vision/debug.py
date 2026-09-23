"""Debug views of the tracking result.

Until the timeline UI exists in weeks 9-10, these are the only way to see what
the pipeline decided. Two views, matching the two axes a user confirms along:

    render_overlay()   spatial -- which box is which object, drawn on frames
    format_timeline()  temporal -- when each object is present, as text

A gap in the text timeline is the same signal the UI will eventually show as a
`track_gap` warning, so mistakes surface here first.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from cutpilot.schemas.track import Track, TrackBoxes

PRESENT = "█"
ABSENT = "·"


def color_for(track_id: str) -> tuple[int, int, int]:
    """Stable BGR colour per track id, spread around the hue circle."""
    hue = (hash(track_id) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def render_overlay(
    video_path: Path,
    target: Path,
    tracks: list[Track],
    boxes: dict[str, TrackBoxes],
    *,
    fps: float,
    max_frames: int | None = None,
) -> Path:
    """Write a copy of the video with every tracked box drawn and labelled."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target.parent.mkdir(parents=True, exist_ok=True)

    # cv2.VideoWriter_fourcc is the OpenCV 4 spelling and is deprecated in 5.x.
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open writer for {target}")

    by_id = {t.id: t for t in tracks}
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (max_frames is not None and index >= max_frames):
                break
            t = index / fps

            for track_id, track_boxes in boxes.items():
                track = by_id.get(track_id)
                if track is None or not track.present_at(t):
                    continue
                box = track_boxes.at(t)
                if box is None:
                    continue

                x, y, w, h = box.clamped().scaled_to(width, height)
                color = color_for(track_id)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                label = track.display_name or track_id
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"{t:6.2f}s",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()

    return target


def track_bar(track: Track, duration: float, width: int = 60) -> str:
    """One row of the ASCII timeline for a single track."""
    if duration <= 0 or width <= 0:
        return ""
    cells = []
    for i in range(width):
        t = (i + 0.5) / width * duration
        cells.append(PRESENT if track.present_at(t) else ABSENT)
    return "".join(cells)


def format_timeline(
    tracks: list[Track],
    duration: float,
    *,
    width: int = 60,
    label_width: int = 14,
) -> str:
    """The whole tracking result as text -- what the timeline UI will draw."""
    if not tracks:
        return "(no tracks)"

    lines = []
    ruler = " " * label_width + f"0s{'':<{max(0, width - 8)}}{duration:.0f}s"
    lines.append(ruler)
    lines.append(" " * label_width + "─" * width)

    for track in sorted(tracks, key=lambda t: (t.kind, t.id)):
        bar = track_bar(track, duration, width)
        seen = f"{track.total_duration:5.1f}s"
        gaps = len(track.segments) - 1
        suffix = f"  {seen}" + (f"  gaps:{gaps}" if gaps else "")
        lines.append(f"{track.id:<{label_width}}{bar}{suffix}")

    return "\n".join(lines)
