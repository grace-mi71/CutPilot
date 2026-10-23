"""Turning sightings into presence, and keeping analysis results across turns."""

from __future__ import annotations

import pytest

from cutpilot.schemas import Segment, TimedBox, Track, TrackBoxes
from cutpilot.vision.detect import TrackingResult, segments_from_times

FRAME = 1 / 25


class TestSegmentsFromTimes:
    def test_contiguous_sightings_form_one_segment(self):
        times = [i * FRAME for i in range(10)]
        (segment,) = segments_from_times(times, FRAME)
        assert segment.start == 0.0
        assert segment.end == pytest.approx(10 * FRAME)

    def test_a_short_dropout_does_not_split_the_presence(self):
        """A few missed frames behind someone else is still one presence."""
        times = [0.0, FRAME, 4 * FRAME, 5 * FRAME]
        assert len(segments_from_times(times, FRAME, max_gap_frames=5)) == 1

    def test_a_long_absence_splits_it(self):
        times = [0.0, FRAME, 2.0, 2.0 + FRAME]
        segments = segments_from_times(times, FRAME, max_gap_frames=5)
        assert [s.start for s in segments] == [0.0, 2.0]

    def test_a_single_sighting_still_has_duration(self):
        (segment,) = segments_from_times([1.0], FRAME)
        assert segment.duration == pytest.approx(FRAME)

    def test_order_of_input_does_not_matter(self):
        assert segments_from_times([0.08, 0.0, 0.04], FRAME) == segments_from_times(
            [0.0, 0.04, 0.08], FRAME
        )

    def test_nothing_seen_is_no_segments(self):
        assert segments_from_times([], FRAME) == []


def test_tracking_result_round_trips_through_disk(tmp_path):
    original = TrackingResult(
        tracks=[
            Track(
                id="person_01",
                kind="person",
                display_name="검은 모자를 쓴 사람",
                segments=[Segment(start=0.0, end=2.0)],
            )
        ],
        boxes={
            "person_01": TrackBoxes(
                track_id="person_01", boxes=[TimedBox(t=0.0, x=0.1, y=0.2, w=0.3, h=0.4)]
            )
        },
        frames_processed=50,
        detections_total=50,
        dropped_tracks=2,
    )
    loaded = TrackingResult.load(original.save(tmp_path / "tracks.json"))
    assert loaded.tracks == original.tracks
    assert loaded.boxes == original.boxes
    assert loaded.dropped_tracks == 2
