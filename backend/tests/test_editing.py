"""Effects and reframe geometry -- the parts of the executor that can be
checked without encoding a video."""

from __future__ import annotations

import numpy as np
import pytest

from cutpilot.editing import effects, reframe
from cutpilot.editing.executor import _crop_origin, _even
from cutpilot.schemas.track import Box, TimedBox, TrackBoxes


@pytest.fixture
def noisy_frame() -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 256, size=(200, 320, 3), dtype=np.uint8)


class TestEffects:
    def test_blur_changes_only_its_region(self, noisy_frame):
        before = noisy_frame.copy()
        assert effects.blur_region(noisy_frame, 50, 50, 60, 60, strength=40)
        assert not np.array_equal(noisy_frame[50:110, 50:110], before[50:110, 50:110])
        assert np.array_equal(noisy_frame[0:40, 0:40], before[0:40, 0:40])

    def test_mosaic_collapses_the_region_to_few_colours(self, noisy_frame):
        effects.mosaic_region(noisy_frame, 40, 40, 80, 80, strength=20)
        region = noisy_frame[40:120, 40:120].reshape(-1, 3)
        # strength 20 -> 5 blocks per side -> at most 25 distinct colours.
        assert len({tuple(px) for px in region}) <= 25

    def test_stronger_mosaic_leaves_fewer_colours(self, noisy_frame):
        mild = noisy_frame.copy()
        harsh = noisy_frame.copy()
        effects.mosaic_region(mild, 40, 40, 80, 80, strength=5)
        effects.mosaic_region(harsh, 40, 40, 80, 80, strength=50)

        def colours(frame):
            return len({tuple(px) for px in frame[40:120, 40:120].reshape(-1, 3)})

        assert colours(harsh) < colours(mild)

    def test_region_smaller_than_the_floor_is_left_alone(self, noisy_frame):
        before = noisy_frame.copy()
        assert not effects.blur_region(noisy_frame, 10, 10, 2, 2, strength=40)
        assert np.array_equal(noisy_frame, before)

    def test_region_hanging_off_the_edge_is_clipped(self, noisy_frame):
        """Detector boxes routinely run past the frame; this must not raise."""
        assert effects.blur_region(noisy_frame, 300, 180, 80, 80, strength=40)
        assert effects.mosaic_region(noisy_frame, -20, -20, 60, 60, strength=20)

    def test_region_entirely_outside_the_frame_does_nothing(self, noisy_frame):
        before = noisy_frame.copy()
        assert not effects.blur_region(noisy_frame, 400, 400, 50, 50, strength=40)
        assert np.array_equal(noisy_frame, before)

    def test_strength_is_clamped_into_range(self):
        assert effects.clamp_strength(0, 25) == effects.MIN_STRENGTH
        assert effects.clamp_strength(900, 25) == effects.MAX_STRENGTH
        assert effects.clamp_strength(None, 25) == 25

    def test_reframe_is_not_a_pixel_effect(self, noisy_frame):
        with pytest.raises(ValueError, match="not a privacy effect"):
            effects.apply_effect(noisy_frame, "REFRAME", 0, 0, 10, 10, 20)


class TestAspectRatio:
    def test_parses_a_ratio(self):
        assert reframe.parse_aspect_ratio("9:16") == pytest.approx(0.5625)
        assert reframe.parse_aspect_ratio("16:9") == pytest.approx(1.7778, abs=1e-4)

    def test_parses_a_bare_float(self):
        assert reframe.parse_aspect_ratio("1.5") == pytest.approx(1.5)

    @pytest.mark.parametrize("value", ["0:16", "9:0", "-1"])
    def test_rejects_nonsense(self, value):
        with pytest.raises(ValueError, match="bad aspect ratio"):
            reframe.parse_aspect_ratio(value)


class TestCropSize:
    def test_landscape_to_portrait_loses_width(self):
        """The headline conversion: 1920x1080 -> 9:16 keeps full height."""
        size = reframe.crop_size_for(1920, 1080, reframe.parse_aspect_ratio("9:16"))
        assert size.h == pytest.approx(1.0)
        assert size.w == pytest.approx(0.31640625)
        assert size.moves_horizontally

    def test_portrait_to_landscape_loses_height(self):
        size = reframe.crop_size_for(1080, 1920, reframe.parse_aspect_ratio("16:9"))
        assert size.w == pytest.approx(1.0)
        assert not size.moves_horizontally

    def test_matching_ratio_crops_nothing(self):
        """The sample clip is already 9:16 -- reframing it to 9:16 is a no-op,
        which is why Auto Reframe cannot be exercised on that footage."""
        size = reframe.crop_size_for(1080, 1920, reframe.parse_aspect_ratio("9:16"))
        assert size.w == pytest.approx(1.0)
        assert size.h == pytest.approx(1.0)


class TestWindowPlacement:
    SIZE = reframe.CropSize(w=0.3, h=1.0)

    def test_centres_on_the_subject(self):
        window = reframe.window_at((0.5, 0.5), self.SIZE)
        assert window.x == pytest.approx(0.35)

    def test_clamps_at_the_left_edge(self):
        assert reframe.window_at((0.02, 0.5), self.SIZE).x == 0.0

    def test_clamps_at_the_right_edge(self):
        window = reframe.window_at((0.98, 0.5), self.SIZE)
        assert window.right == pytest.approx(1.0)


class TestSmoothing:
    def test_holds_position_through_a_gap(self):
        """A dropped detection must not yank the crop to a default and back."""
        centres = [(0.3, 0.5), None, None, (0.3, 0.5)]
        assert all(c[0] == pytest.approx(0.3) for c in reframe.smooth(centres, fps=4.0))

    def test_leading_gap_borrows_the_first_known_centre(self):
        centres = [None, None, (0.8, 0.5)]
        assert reframe.smooth(centres, fps=3.0)[0][0] == pytest.approx(0.8)

    def test_averaging_reduces_jitter(self):
        jittery = [(0.5 + (0.1 if i % 2 else -0.1), 0.5) for i in range(40)]
        smoothed = reframe.smooth(jittery, fps=20.0, window_seconds=1.0)
        spread = max(c[0] for c in smoothed) - min(c[0] for c in smoothed)
        assert spread < 0.05

    def test_zero_window_passes_values_through(self):
        centres = [(0.1, 0.5), (0.9, 0.5)]
        assert reframe.smooth(centres, fps=10.0, window_seconds=0.0) == centres

    def test_empty_input_is_empty_output(self):
        assert reframe.smooth([], fps=25.0) == []


class TestTrajectory:
    def test_one_window_per_frame(self):
        boxes = TrackBoxes(
            track_id="person_01",
            boxes=[TimedBox(t=i / 25, x=0.4, y=0.3, w=0.1, h=0.4) for i in range(25)],
        )
        windows = reframe.crop_trajectory(
            boxes,
            source_w=1920,
            source_h=1080,
            target_ratio=0.5625,
            duration=1.0,
            fps=25.0,
        )
        assert len(windows) == 25
        assert all(w.h == pytest.approx(1.0) for w in windows)


class TestOutputGeometry:
    def test_odd_dimensions_are_rounded_down(self):
        """H.264 rejects odd sizes."""
        assert _even(607) == 606
        assert _even(608) == 608

    def test_crop_origin_stays_inside_the_frame(self):
        at_edge = Box(x=0.95, y=0.0, w=0.3, h=1.0)
        x, y = _crop_origin(at_edge, 1920, 1080, 608, 1080)
        assert 0 <= x <= 1920 - 608
        assert y == 0
