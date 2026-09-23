"""Detection and tracking, producing the Track records the timeline reads.

Tracking runs through ultralytics rather than supervision's ByteTrack wrapper,
which is deprecated and disappears in supervision 0.31. The practical win is the
escalation path recorded in docs/decisions/0001: ByteTrack first, BoT-SORT with
re-identification if identities start swapping. Here that is a config filename,
not a rewrite.

Output follows the split in schemas.track: presence segments on the Track, and
per-frame geometry in TrackBoxes, fetched separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cutpilot.schemas.track import ObjectKind, Segment, TimedBox, Track, TrackBoxes

TRACKER_BYTETRACK = "bytetrack.yaml"
TRACKER_BOTSORT = "botsort.yaml"
"""Risk 2 fallback: BoT-SORT carries appearance embeddings, so it survives the
crossings and re-entries that make ByteTrack hand out a fresh id."""

COCO_PERSON_CLASS = 0

# A track that vanishes for a few frames and comes back is the same presence,
# not two. Wider than this and the gap is real enough to show the user, who is
# the only one who knows whether the subject left or the tracker lost them.
DEFAULT_GAP_FRAMES = 5

# Trackers hand out an id to anything they see, including a blur that looked
# like a person for two frames. Those ids are not editable objects -- nobody can
# point at them, confirm them, or blur them usefully -- but they would crowd the
# object list and, worse, inflate the ids the agent reasons over.
DEFAULT_MIN_DURATION = 0.5


@dataclass
class TrackingResult:
    tracks: list[Track] = field(default_factory=list)
    boxes: dict[str, TrackBoxes] = field(default_factory=dict)
    frames_processed: int = 0
    detections_total: int = 0
    dropped_tracks: int = 0
    """Ids the tracker issued that were too brief to be editable objects."""

    def by_id(self) -> dict[str, Track]:
        return {t.id: t for t in self.tracks}


def segments_from_times(
    times: list[float],
    frame_interval: float,
    *,
    max_gap_frames: int = DEFAULT_GAP_FRAMES,
) -> list[Segment]:
    """Collapse the instants an object was seen into presence intervals.

    Intervals are half-open and extend one frame past the last sighting, so a
    single-frame detection still has non-zero duration.
    """
    if not times:
        return []

    ordered = sorted(times)
    tolerance = frame_interval * max_gap_frames
    segments: list[Segment] = []
    start = previous = ordered[0]

    for t in ordered[1:]:
        if t - previous > tolerance:
            segments.append(Segment(start=start, end=previous + frame_interval))
            start = t
        previous = t
    segments.append(Segment(start=start, end=previous + frame_interval))
    return segments


def detect_and_track(
    video_path: Path,
    *,
    model_path: str,
    fps: float,
    kind: ObjectKind = "person",
    device: str = "cpu",
    confidence: float = 0.35,
    classes: list[int] | None = None,
    tracker: str = TRACKER_BYTETRACK,
    max_frames: int | None = None,
    id_prefix: str | None = None,
    min_duration: float = DEFAULT_MIN_DURATION,
) -> TrackingResult:
    """Run detection plus tracking over a whole clip.

    Coordinates come back normalized, matching the convention the rest of the
    backend uses; no pixel values cross this boundary.

    Tracks shorter than `min_duration` are dropped, and the survivors are
    numbered from one in order of appearance. The tracker's own ids are not
    kept: they run well past the number of objects actually present (a 15-person
    clip produced ids into the thirties), and those ids end up in the object
    list and in the agent's prompt, where "person_34" implies a crowd that is
    not there.
    """
    from ultralytics import YOLO  # imported lazily: pulls in torch

    if fps <= 0:
        raise ValueError("fps must be positive")

    model = YOLO(model_path)
    frame_interval = 1.0 / fps
    prefix = id_prefix or kind

    times: dict[int, list[float]] = {}
    boxes: dict[int, list[TimedBox]] = {}
    frames = 0
    detections = 0

    stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=tracker,
        conf=confidence,
        device=device,
        classes=classes if classes is not None else [COCO_PERSON_CLASS],
        verbose=False,
    )

    for index, result in enumerate(stream):
        if max_frames is not None and index >= max_frames:
            break
        frames += 1
        t = index * frame_interval

        detected = result.boxes
        if detected is None or detected.id is None:
            continue

        # xyxyn is already normalized to the frame.
        for (x1, y1, x2, y2), raw_id in zip(
            detected.xyxyn.tolist(), detected.id.tolist(), strict=False
        ):
            track_id = int(raw_id)
            detections += 1
            times.setdefault(track_id, []).append(t)
            boxes.setdefault(track_id, []).append(
                TimedBox(t=t, x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1))
            )

    result_out = TrackingResult(frames_processed=frames, detections_total=detections)

    surviving: list[tuple[float, list[Segment], list[TimedBox]]] = []
    for track_id in sorted(times):
        segments = segments_from_times(times[track_id], frame_interval)
        if sum(s.duration for s in segments) < min_duration:
            result_out.dropped_tracks += 1
            continue
        surviving.append((segments[0].start, segments, boxes[track_id]))

    # Sort on first appearance only; Segment lists are not orderable.
    surviving.sort(key=lambda item: item[0])
    for number, (_, segments, track_boxes) in enumerate(surviving, start=1):
        name = f"{prefix}_{number:02d}"
        result_out.tracks.append(Track(id=name, kind=kind, segments=segments))
        result_out.boxes[name] = TrackBoxes(track_id=name, boxes=track_boxes)

    return result_out


def merge(*results: TrackingResult) -> TrackingResult:
    """Combine runs over different object kinds into one set of tracks."""
    merged = TrackingResult()
    for result in results:
        merged.tracks.extend(result.tracks)
        merged.boxes.update(result.boxes)
        merged.frames_processed = max(merged.frames_processed, result.frames_processed)
        merged.detections_total += result.detections_total
    return merged
