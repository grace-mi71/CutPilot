"""Geometry and interval arithmetic that the warnings and the renderer rest on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cutpilot.schemas import (
    Box,
    EditPlan,
    Operation,
    Segment,
    Selector,
    TimedBox,
    TimeRange,
    Track,
    TrackBoxes,
)


class TestBox:
    def test_iou_of_identical_boxes_is_one(self):
        box = Box(x=0.1, y=0.1, w=0.2, h=0.2)
        assert box.iou(box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_no_overlap(self):
        a = Box(x=0.0, y=0.0, w=0.1, h=0.1)
        b = Box(x=0.5, y=0.5, w=0.1, h=0.1)
        assert a.iou(b) == 0.0
        assert a.containment_in(b) == 0.0

    def test_face_inside_person_scores_high_containment_but_low_iou(self):
        """Why linking uses containment: a face is fully inside its person, yet
        IoU between them is small because the person box dwarfs it."""
        person = Box(x=0.30, y=0.10, w=0.30, h=0.80)
        face = Box(x=0.40, y=0.12, w=0.08, h=0.10)
        assert face.containment_in(person) == pytest.approx(1.0)
        assert face.iou(person) < 0.05

    def test_clamped_pulls_box_inside_the_frame(self):
        clamped = Box(x=-0.1, y=0.9, w=0.5, h=0.5).clamped()
        assert clamped.x == 0.0
        assert clamped.y == pytest.approx(0.9)
        assert clamped.bottom <= 1.0

    def test_scaled_to_pixels(self):
        assert Box(x=0.5, y=0.5, w=0.25, h=0.5).scaled_to(1920, 1080) == (960, 540, 480, 540)


class TestSegment:
    def test_overlap_returns_intersection(self):
        overlap = Segment(start=0.0, end=10.0).overlap(5.0, 20.0)
        assert overlap == Segment(start=5.0, end=10.0)

    def test_touching_segments_do_not_overlap(self):
        assert Segment(start=0.0, end=10.0).overlap(10.0, 20.0) is None


class TestTrackGaps:
    """gaps_within drives the track_gap warning -- an unflagged gap in a blur
    target is an uncovered face in the delivered video."""

    @pytest.fixture
    def interrupted(self) -> Track:
        return Track(
            id="person_02",
            kind="person",
            segments=[Segment(start=0.0, end=12.0), Segment(start=18.0, end=30.0)],
        )

    def test_finds_the_interior_gap(self, interrupted):
        assert interrupted.gaps_within(0.0, 30.0) == [Segment(start=12.0, end=18.0)]

    def test_reports_no_gap_when_window_sits_inside_a_segment(self, interrupted):
        assert interrupted.gaps_within(2.0, 8.0) == []

    def test_partial_overlap_reports_only_the_missing_part(self, interrupted):
        assert interrupted.gaps_within(10.0, 20.0) == [Segment(start=12.0, end=18.0)]

    def test_window_past_the_end_is_entirely_a_gap(self, interrupted):
        assert interrupted.gaps_within(30.0, 40.0) == [Segment(start=30.0, end=40.0)]

    def test_track_never_seen_is_one_long_gap(self):
        assert Track(id="x", kind="face").gaps_within(0.0, 5.0) == [Segment(start=0.0, end=5.0)]

    def test_present_at_respects_segment_boundaries(self, interrupted):
        assert interrupted.present_at(11.9)
        assert not interrupted.present_at(12.0)
        assert not interrupted.present_at(15.0)


class TestTrackBoxes:
    @pytest.fixture
    def boxes(self) -> TrackBoxes:
        """Samples land a few frames apart, as they do coming out of a tracker.

        The spacing matters: `max_gap` is the widest stride `at()` will
        interpolate across, so a fixture with second-wide steps would look like
        a detection dropout rather than normal sampling.
        """
        return TrackBoxes(
            track_id="person_02",
            boxes=[
                TimedBox(t=0.0, x=0.0, y=0.0, w=0.2, h=0.4),
                TimedBox(t=0.2, x=0.4, y=0.0, w=0.2, h=0.4),
            ],
        )

    def test_exact_sample_is_returned_as_is(self, boxes):
        assert boxes.at(0.2) == Box(x=0.4, y=0.0, w=0.2, h=0.4)

    def test_midpoint_is_interpolated(self, boxes):
        assert boxes.at(0.1).x == pytest.approx(0.2)

    def test_stride_wider_than_tolerance_is_not_interpolated(self):
        """Two samples a second apart mean the object went undetected between
        them; inventing a position there would slide a blur off its subject."""
        dropout = TrackBoxes(
            track_id="t",
            boxes=[
                TimedBox(t=0.0, x=0.0, y=0.0, w=0.1, h=0.1),
                TimedBox(t=1.0, x=0.4, y=0.0, w=0.1, h=0.1),
            ],
        )
        assert dropout.at(0.5) is None

    def test_gap_wider_than_tolerance_yields_nothing(self):
        """A blur must not drift across frames where the object was not seen."""
        sparse = TrackBoxes(
            track_id="t",
            boxes=[
                TimedBox(t=0.0, x=0.0, y=0.0, w=0.1, h=0.1),
                TimedBox(t=5.0, x=0.9, y=0.0, w=0.1, h=0.1),
            ],
        )
        assert sparse.at(2.5, max_gap=0.5) is None

    def test_time_beyond_the_last_sample_is_dropped(self, boxes):
        assert boxes.at(9.0) is None

    def test_empty_track_returns_nothing(self):
        assert TrackBoxes(track_id="t").at(0.0) is None


class TestTimeRange:
    def test_none_bounds_mean_the_whole_video(self):
        assert TimeRange().resolve(30.0) == (0.0, 30.0)

    def test_end_is_clipped_to_duration(self):
        assert TimeRange(start=10.0, end=99.0).resolve(30.0) == (10.0, 30.0)

    def test_intersect_returns_none_when_disjoint(self):
        assert TimeRange(start=0.0, end=5.0).intersect(10.0, 20.0, 30.0) is None

    def test_describe_names_the_full_span(self):
        assert TimeRange().describe(30.0) == "전체 구간"
        assert TimeRange(start=10.0, end=25.0).describe(30.0) == "10.0s ─ 25.0s"


class TestOperation:
    def test_reframe_requires_an_aspect_ratio(self):
        with pytest.raises(ValidationError, match="aspect_ratio"):
            Operation(op="REFRAME", selector=Selector(kind="person", include=["person_02"]))

    def test_reframe_follows_exactly_one_subject(self):
        with pytest.raises(ValidationError, match="one subject"):
            Operation(
                op="REFRAME",
                selector=Selector(kind="person", include=["person_02", "person_03"]),
                aspect_ratio="9:16",
            )

    def test_blur_defaults_to_whole_kind_and_whole_video(self):
        op = Operation(op="BLUR", selector=Selector(kind="face"))
        assert op.selector.is_whole_kind
        assert op.region == "bbox"
        assert op.time_range.resolve(30.0) == (0.0, 30.0)


class TestEditPlan:
    def test_splits_reframe_from_privacy_ops(self):
        plan = EditPlan(
            video_id="v1",
            operations=[
                Operation(
                    op="REFRAME",
                    selector=Selector(kind="person", include=["person_02"]),
                    aspect_ratio="9:16",
                ),
                Operation(
                    op="MOSAIC",
                    selector=Selector(kind="face", exclude=["face_02"]),
                    time_range=TimeRange(start=10.0, end=25.0),
                ),
            ],
        )
        assert plan.reframe is not None
        assert len(plan.privacy_ops) == 1
        assert plan.referenced_track_ids() == {"person_02", "face_02"}
