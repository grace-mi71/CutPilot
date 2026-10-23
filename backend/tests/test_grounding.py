"""Grounding and the confirmation it produces.

Scene: three people side by side on a landscape frame, two of them with
faces, one with a gap in the middle of their track.

    person_01  left    face_01
    person_02  centre
    person_03  right   face_03   absent 4-6s
"""

from __future__ import annotations

import re

import pytest

from cutpilot.agent.confirm import build_confirmation, ground_draft
from cutpilot.agent.grounder import (
    DetectedObject,
    ground_phrase,
    is_unresolved,
    match,
    pick_keyframe,
    to_box,
)
from cutpilot.agent.llm import FakeProvider
from cutpilot.schemas import (
    Box,
    EditPlanDraft,
    OperationDraft,
    Segment,
    SelectorDraft,
    TimedBox,
    TimeRange,
    Track,
    TrackBoxes,
    VideoAsset,
)
from cutpilot.schemas.session import UIContext
from cutpilot.vision.resolve import link_children, resolve_selector

DURATION = 10.0
STEP = 0.1

GEOMETRY = {
    "person_01": (0.05, 0.10, 0.15, 0.80),
    "person_02": (0.42, 0.10, 0.15, 0.80),
    "person_03": (0.80, 0.10, 0.15, 0.80),
    "face_01": (0.09, 0.12, 0.06, 0.10),
    "face_03": (0.84, 0.12, 0.06, 0.10),
}
GAPS = {"person_03": (4.0, 6.0), "face_03": (4.0, 6.0)}


def _presence(track_id: str) -> list[Segment]:
    if track_id in GAPS:
        lo, hi = GAPS[track_id]
        return [Segment(start=0.0, end=lo), Segment(start=hi, end=DURATION)]
    return [Segment(start=0.0, end=DURATION)]


def _boxes(track_id: str) -> TrackBoxes:
    x, y, w, h = GEOMETRY[track_id]
    samples = []
    for i in range(int(DURATION / STEP)):
        t = round(i * STEP, 3)
        if any(s.contains(t) for s in _presence(track_id)):
            samples.append(TimedBox(t=t, x=x, y=y, w=w, h=h))
    return TrackBoxes(track_id=track_id, boxes=samples)


@pytest.fixture
def boxes() -> dict[str, TrackBoxes]:
    return {track_id: _boxes(track_id) for track_id in GEOMETRY}


@pytest.fixture
def tracks(boxes) -> list[Track]:
    raw = [
        Track(id=tid, kind="face" if tid.startswith("face") else "person", segments=_presence(tid))
        for tid in GEOMETRY
    ]
    return link_children(raw, boxes)


@pytest.fixture
def asset() -> VideoAsset:
    return VideoAsset(
        id="scene", source_path="s.mp4", proxy_path="p.mp4", sprite_path="sp.jpg",
        duration=DURATION, fps=10.0, width=1920, height=1080,
    )  # fmt: skip


def _gemini_box(track_id: str, nudge: float = 0.01) -> list[int]:
    """What Gemini would return for a track: approximately its box, 0-1000."""
    x, y, w, h = GEOMETRY[track_id]
    return [
        int((y + nudge) * 1000),
        int((x + nudge) * 1000),
        int((y + h) * 1000),
        int((x + w) * 1000),
    ]


def detector(mapping: dict[str, list[str]]):
    """A FakeProvider that 'sees' fixed people for each phrase."""

    def respond(request):
        phrase = re.search(r'"(.+?)"', request.prompt).group(1)
        ids = mapping.get(phrase, [])
        return {"objects": [{"box_2d": _gemini_box(i), "label": phrase} for i in ids]}

    return FakeProvider(responder=respond)


FRAME = lambda _t: b"jpeg"  # noqa: E731


class TestBoxConversion:
    def test_gemini_order_is_y_first(self):
        box = to_box(DetectedObject(box_2d=[100, 200, 500, 400]))
        assert (box.x, box.y, box.w, box.h) == pytest.approx((0.2, 0.1, 0.2, 0.4))

    def test_inverted_corners_are_repaired(self):
        box = to_box(DetectedObject(box_2d=[500, 400, 100, 200]))
        assert box.w > 0 and box.h > 0


class TestKeyframe:
    def test_uses_the_playhead_when_it_is_in_range(self, tracks, boxes):
        t = pick_keyframe(
            TimeRange(start=1.0, end=9.0), duration=DURATION, kind="person",
            tracks=tracks, boxes=boxes, ui=UIContext(playhead=3.0),
        )  # fmt: skip
        assert t == 3.0

    def test_falls_back_to_the_start_of_the_range(self, tracks, boxes):
        t = pick_keyframe(
            TimeRange(start=2.0, end=9.0), duration=DURATION, kind="person",
            tracks=tracks, boxes=boxes, ui=UIContext(playhead=50.0),
        )  # fmt: skip
        assert t == 2.0


class TestMatch:
    def test_box_between_two_people_is_ambiguous(self, tracks, boxes):
        present = [(t, boxes[t.id].at(1.0)) for t in tracks if t.kind == "person"]
        straddling = Box(x=0.05, y=0.10, w=0.52, h=0.80)  # covers 01 and 02 equally
        candidates, ambiguous = match([straddling], present)
        assert ambiguous or not candidates


class TestGroundPhrase:
    def _ground(self, provider, phrase, tracks, boxes, kind="face"):
        return ground_phrase(
            provider, phrase, kind=kind, time_range=TimeRange(), duration=DURATION,
            tracks=tracks, boxes=boxes, frame_loader=FRAME,
        )  # fmt: skip

    def test_a_track_id_needs_no_model_call(self, tracks, boxes):
        fake = FakeProvider()
        result = self._ground(fake, "@person_02", tracks, boxes)
        assert (result.status, result.chosen) == ("ok", ["person_02"])
        assert fake.requests == []

    def test_an_unknown_id_is_a_choice_not_a_guess(self, tracks, boxes):
        result = self._ground(FakeProvider(), "person_99", tracks, boxes)
        assert result.status == "choose" and result.chosen == []

    def test_one_clear_match_asks_for_confirmation(self, tracks, boxes):
        fake = detector({"검은 모자 쓴 사람": ["person_03"]})
        result = self._ground(fake, "검은 모자 쓴 사람", tracks, boxes)
        assert (result.status, result.chosen) == ("confirm", ["person_03"])

    def test_appearance_is_matched_against_people_even_for_faces(self, tracks, boxes):
        """The cap is on the person box. Faces are reached through parents."""
        fake = detector({"검은 모자 쓴 사람": ["person_03"]})
        result = self._ground(fake, "검은 모자 쓴 사람", tracks, boxes, kind="face")
        assert result.kind == "person"
        assert fake.last.thinking_level == "minimal"

    def test_several_matches_mean_choose(self, tracks, boxes):
        fake = detector({"검은 옷": ["person_01", "person_03"]})
        result = self._ground(fake, "검은 옷", tracks, boxes)
        assert result.status == "choose"
        assert {c.track_id for c in result.candidates} == {"person_01", "person_03"}

    def test_no_match_offers_everyone_on_screen(self, tracks, boxes):
        result = self._ground(detector({}), "빨간 우산", tracks, boxes)
        assert result.status == "choose"
        assert result.chosen == []
        assert {c.track_id for c in result.candidates} == {"person_01", "person_02", "person_03"}


def _op(op="MOSAIC", kind="face", include=(), exclude=(), ratio=None, start=None, end=None):
    return OperationDraft(
        op=op,
        selector=SelectorDraft(
            kind=kind, include_phrases=list(include), exclude_phrases=list(exclude)
        ),
        time_range=TimeRange(start=start, end=end),
        aspect_ratio=ratio,
    )


def _ground_draft(provider, ops, asset, tracks, boxes, ui=None):
    return ground_draft(
        provider, EditPlanDraft(operations=list(ops)),
        asset=asset, tracks=tracks, boxes=boxes, frame_loader=FRAME, ui=ui,
    )  # fmt: skip


class TestGroundDraft:
    def test_an_unresolved_target_never_widens_to_everyone(self, asset, tracks, boxes):
        """The trap: an include that grounds to nothing must not become an
        empty include, which would mean every face in the video."""
        result = _ground_draft(
            detector({}), [_op(include=["빨간 우산 든 사람"])], asset, tracks, boxes
        )
        selector = result.plan.operations[0].selector
        assert selector.include and all(is_unresolved(i) for i in selector.include)
        assert resolve_selector(selector, tracks) == []
        assert any(w.kind == "empty_selection" for w in result.confirmation.warnings)
        assert result.confirmation.blocking_items

    def test_sparing_a_person_by_description_spares_their_face(self, asset, tracks, boxes):
        fake = detector({"왼쪽 남자": ["person_01"]})
        result = _ground_draft(fake, [_op(exclude=["왼쪽 남자"])], asset, tracks, boxes)
        covered = {t.id for t in resolve_selector(result.plan.operations[0].selector, tracks)}
        assert covered == {"face_03"}
        roles = {(o.track_id, o.role) for o in result.confirmation.overlays}
        assert ("person_01", "excluded") in roles

    def test_reframe_without_a_subject_asks_who(self, asset, tracks, boxes):
        result = _ground_draft(
            FakeProvider(), [_op(op="REFRAME", kind="person", ratio="9:16")], asset, tracks, boxes
        )
        assert len(result.plan.operations[0].selector.include) == 1
        selector_item = next(i for i in result.confirmation.items if i.field == "selector")
        assert selector_item.status == "choose"
        assert selector_item.candidates

    def test_ids_only_plan_needs_no_input(self, asset, tracks, boxes):
        result = _ground_draft(FakeProvider(), [_op(include=["person_02"])], asset, tracks, boxes)
        assert not result.confirmation.needs_input


class TestWarnings:
    def test_gap_in_an_explicit_target_is_flagged(self, asset, tracks, boxes):
        result = _ground_draft(FakeProvider(), [_op(include=["person_03"])], asset, tracks, boxes)
        gaps = [w for w in result.confirmation.warnings if w.kind == "track_gap"]
        assert gaps and gaps[0].time_range.start == pytest.approx(4.0, abs=0.15)
        assert gaps[0].is_privacy_risk

    def test_gaps_are_not_flagged_for_whole_kind_selections(self, asset, tracks, boxes):
        result = _ground_draft(FakeProvider(), [_op()], asset, tracks, boxes)
        assert not any(w.kind == "track_gap" for w in result.confirmation.warnings)

    def test_blur_cropped_away_by_the_reframe_is_flagged(self, asset, tracks, boxes):
        """Reframe onto the left person, mosaic everyone: the right-hand face
        falls outside a 9:16 crop of a 16:9 frame."""
        result = _ground_draft(
            FakeProvider(),
            [_op(op="REFRAME", kind="person", include=["person_01"], ratio="9:16"), _op()],
            asset, tracks, boxes,
        )  # fmt: skip
        assert any(w.kind == "out_of_frame" for w in result.confirmation.warnings)

    def test_rebuild_from_an_edited_plan(self, asset, tracks, boxes):
        """The confirmation can be rebuilt after the user edits the plan."""
        first = _ground_draft(FakeProvider(), [_op(include=["person_03"])], asset, tracks, boxes)
        again = build_confirmation(first.plan, {}, asset=asset, tracks=tracks, boxes=boxes)
        assert [i.field for i in again.items] == ["op", "selector", "time_range"]
