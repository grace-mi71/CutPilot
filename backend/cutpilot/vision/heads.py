"""Face regions estimated from person boxes, until a face detector is in place.

"Blur his face" is the most common privacy request, and the planner reads
"blur that person" as a face edit more often than not. Without face tracks
every such request selects nobody. This module gives each person a face track
whose box is the head region of the person box -- the top of the box, centred,
generously sized, because for privacy covering a little too much is the safe
side of the error. It is switched by CUTPILOT_ESTIMATE_FACES.

It is an estimate, not a detection: it covers the back of a head as readily as
a face, and a person bending or sitting gets a poorer box. A real detector
(YuNet) produces face tracks of exactly this shape -- kind "face", parent_id
the person -- and replaces this without anything downstream changing.
"""

from __future__ import annotations

from cutpilot.schemas.track import Box, TimedBox, Track, TrackBoxes
from cutpilot.vision.detect import TrackingResult

HEAD_WIDTH = 0.6
"""Of the person box width. Person boxes include swinging arms and bags."""

HEAD_HEIGHT = 0.22
"""Of the person box height, from the top."""


def head_box(person: Box) -> Box:
    w = person.w * HEAD_WIDTH
    return Box(x=person.x + (person.w - w) / 2, y=person.y, w=w, h=person.h * HEAD_HEIGHT).clamped()


def face_id_for(person_id: str) -> str:
    return person_id.replace("person_", "face_", 1)


def with_head_regions(result: TrackingResult) -> TrackingResult:
    """Add one estimated face track per person. A no-op if faces already exist,
    so a real detector's output is never overwritten."""
    if any(t.kind == "face" for t in result.tracks):
        return result
    tracks = list(result.tracks)
    boxes = dict(result.boxes)
    for person in result.tracks:
        if person.kind != "person" or person.id not in result.boxes:
            continue
        face_id = face_id_for(person.id)
        tracks.append(
            Track(id=face_id, kind="face", parent_id=person.id, segments=list(person.segments))
        )
        boxes[face_id] = TrackBoxes(
            track_id=face_id,
            boxes=[
                TimedBox(t=b.t, **head_box(b).model_dump()) for b in result.boxes[person.id].boxes
            ],
        )
    return TrackingResult(
        tracks=tracks,
        boxes=boxes,
        frames_processed=result.frames_processed,
        detections_total=result.detections_total,
        dropped_tracks=result.dropped_tracks,
    )
