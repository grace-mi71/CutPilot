"""Selector resolution -- the step that carries a reference from what the user
can describe to what the edit has to cover.

Scene used throughout: two people, each with a face, plus a plate.

    person_02  left      face_02
    person_03  right     face_03   (the one in the black cap)
    plate_01   bottom
"""

from __future__ import annotations

import pytest

from cutpilot.schemas import Segment, Selector, TimedBox, Track, TrackBoxes
from cutpilot.vision.resolve import (
    link_children,
    resolve_region_box,
    resolve_selector,
    translate_to_kind,
)


def _boxes(track_id: str, x: float, y: float, w: float, h: float) -> TrackBoxes:
    return TrackBoxes(
        track_id=track_id,
        boxes=[TimedBox(t=float(t), x=x, y=y, w=w, h=h) for t in range(4)],
    )


@pytest.fixture
def scene_boxes() -> dict[str, TrackBoxes]:
    return {
        "person_02": _boxes("person_02", 0.05, 0.10, 0.30, 0.80),
        "person_03": _boxes("person_03", 0.55, 0.10, 0.30, 0.80),
        "face_02": _boxes("face_02", 0.15, 0.12, 0.08, 0.10),
        "face_03": _boxes("face_03", 0.65, 0.12, 0.08, 0.10),
        "plate_01": _boxes("plate_01", 0.40, 0.80, 0.12, 0.06),
    }


@pytest.fixture
def raw_tracks() -> list[Track]:
    span = [Segment(start=0.0, end=4.0)]
    return [
        Track(id="person_02", kind="person", segments=span),
        Track(id="person_03", kind="person", display_name="검은 모자를 쓴 사람", segments=span),
        Track(id="face_02", kind="face", segments=span),
        Track(id="face_03", kind="face", segments=span),
        Track(id="plate_01", kind="license_plate", segments=span),
    ]


@pytest.fixture
def tracks(raw_tracks, scene_boxes) -> list[Track]:
    return link_children(raw_tracks, scene_boxes)


class TestLinkChildren:
    def test_each_face_attaches_to_the_person_containing_it(self, tracks):
        by_id = {t.id: t for t in tracks}
        assert by_id["face_02"].parent_id == "person_02"
        assert by_id["face_03"].parent_id == "person_03"

    def test_people_have_no_parent(self, tracks):
        assert all(t.parent_id is None for t in tracks if t.kind == "person")

    def test_plates_stay_unattached_in_the_mvp(self, tracks):
        assert next(t for t in tracks if t.id == "plate_01").parent_id is None

    def test_face_outside_every_person_is_left_unlinked(self, raw_tracks, scene_boxes):
        orphan = Track(id="face_99", kind="face", segments=[Segment(start=0.0, end=4.0)])
        boxes = {**scene_boxes, "face_99": _boxes("face_99", 0.90, 0.90, 0.05, 0.05)}
        linked = link_children([*raw_tracks, orphan], boxes)
        assert next(t for t in linked if t.id == "face_99").parent_id is None

    def test_linking_does_not_mutate_the_inputs(self, raw_tracks, scene_boxes):
        link_children(raw_tracks, scene_boxes)
        assert all(t.parent_id is None for t in raw_tracks)


class TestTranslateToKind:
    def test_same_kind_passes_through(self, tracks):
        assert translate_to_kind("person_03", "person", tracks) == ["person_03"]

    def test_person_to_face_walks_down_to_the_children(self, tracks):
        """Asking for the black-capped man's face: the cap is only visible on
        the person box, while the mosaic has to land on the face."""
        assert translate_to_kind("person_03", "face", tracks) == ["face_03"]

    def test_face_to_person_walks_up_to_the_owner(self, tracks):
        assert translate_to_kind("face_03", "person", tracks) == ["person_03"]

    def test_unrelated_kind_yields_nothing(self, tracks):
        assert translate_to_kind("person_03", "license_plate", tracks) == []

    def test_unknown_id_yields_nothing_rather_than_raising(self, tracks):
        """A selector that resolves to nothing becomes a visible warning; an
        exception here would just abort the turn."""
        assert translate_to_kind("person_99", "face", tracks) == []


class TestResolveSelector:
    def test_empty_include_means_the_whole_kind(self, tracks):
        picked = resolve_selector(Selector(kind="face"), tracks)
        assert {t.id for t in picked} == {"face_02", "face_03"}

    def test_exclude_by_person_spares_that_persons_face(self, tracks):
        """Sparing the man in black: grounding returns a person, so the
        exclusion has to reach his face."""
        picked = resolve_selector(Selector(kind="face", exclude=["person_02"]), tracks)
        assert {t.id for t in picked} == {"face_03"}

    def test_include_by_person_selects_only_that_persons_face(self, tracks):
        picked = resolve_selector(Selector(kind="face", include=["person_03"]), tracks)
        assert {t.id for t in picked} == {"face_03"}

    def test_include_and_exclude_can_be_combined(self, tracks):
        picked = resolve_selector(
            Selector(kind="face", include=["person_02", "person_03"], exclude=["face_02"]),
            tracks,
        )
        assert {t.id for t in picked} == {"face_03"}

    def test_plates_are_selected_as_a_whole_kind(self, tracks):
        picked = resolve_selector(Selector(kind="license_plate"), tracks)
        assert {t.id for t in picked} == {"plate_01"}

    def test_unresolvable_include_selects_nothing(self, tracks):
        assert resolve_selector(Selector(kind="face", include=["person_99"]), tracks) == []


class TestResolveRegionBox:
    def test_bbox_returns_the_tracks_own_box(self, tracks, scene_boxes):
        track = next(t for t in tracks if t.id == "face_03")
        box = resolve_region_box(track, "bbox", 1.0, tracks, scene_boxes)
        assert box == scene_boxes["face_03"].at(1.0)

    def test_face_region_on_a_person_descends_to_the_face_box(self, tracks, scene_boxes):
        person = next(t for t in tracks if t.id == "person_03")
        box = resolve_region_box(person, "face", 1.0, tracks, scene_boxes)
        assert box == scene_boxes["face_03"].at(1.0)
        assert box != scene_boxes["person_03"].at(1.0)

    def test_missing_boxes_yield_nothing(self, tracks):
        track = next(t for t in tracks if t.id == "face_03")
        assert resolve_region_box(track, "bbox", 1.0, tracks, {}) is None

    def test_body_region_is_reserved(self, tracks, scene_boxes):
        track = next(t for t in tracks if t.id == "person_03")
        with pytest.raises(NotImplementedError):
            resolve_region_box(track, "body", 1.0, tracks, scene_boxes)
