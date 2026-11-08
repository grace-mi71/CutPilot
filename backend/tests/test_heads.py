"""Estimated face tracks: one per person, at the top of the box, never over real faces."""

from __future__ import annotations

from cutpilot.schemas.plan import Selector
from cutpilot.schemas.track import Box, Segment, TimedBox, Track, TrackBoxes
from cutpilot.vision.detect import TrackingResult
from cutpilot.vision.heads import head_box, with_head_regions
from cutpilot.vision.resolve import resolve_selector


def person(track_id: str) -> tuple[Track, TrackBoxes]:
    track = Track(id=track_id, kind="person", segments=[Segment(start=0.0, end=2.0)])
    boxes = TrackBoxes(
        track_id=track_id,
        boxes=[TimedBox(t=t / 10, x=0.2, y=0.1, w=0.2, h=0.6) for t in range(20)],
    )
    return track, boxes


def test_head_sits_at_the_top_centre():
    head = head_box(Box(x=0.2, y=0.1, w=0.2, h=0.6))
    assert abs(head.y - 0.1) < 1e-9 and abs(head.h - 0.132) < 1e-9
    assert abs(head.center[0] - 0.3) < 1e-9


def test_each_person_gets_a_linked_face():
    t1, b1 = person("person_01")
    result = with_head_regions(TrackingResult(tracks=[t1], boxes={"person_01": b1}))
    face = next(t for t in result.tracks if t.kind == "face")
    assert (face.id, face.parent_id) == ("face_01", "person_01")
    assert result.boxes["face_01"].at(1.0) is not None
    # "blur person_01's face" now reaches it
    selector = Selector(kind="face", include=["person_01"])
    assert [t.id for t in resolve_selector(selector, result.tracks)] == ["face_01"]


def test_real_faces_are_left_alone():
    t1, b1 = person("person_01")
    real = Track(id="face_09", kind="face", parent_id="person_01", segments=t1.segments)
    original = TrackingResult(tracks=[t1, real], boxes={"person_01": b1})
    assert with_head_regions(original) is original
