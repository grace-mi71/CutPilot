"""Detection and tracking results.

Coordinates are normalized to 0.0-1.0 everywhere inside the backend. Three
coordinate systems meet in this project -- the source video, the 480p proxy the
UI plays, and the 0-1000 grid the LLM returns -- so pixels are converted once on
the way in and once on the way out, and never in between.

A Track carries only what the timeline needs: which object it is and when it is
on screen. Per-frame boxes live in TrackBoxes and are fetched separately; a
30 fps minute with five objects is ~9,000 boxes, which has no business riding
along in every timeline response.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ObjectKind = Literal["person", "face", "license_plate"]


class Box(BaseModel):
    """Axis-aligned box in normalized coordinates. (x, y) is the top-left."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def clamped(self) -> Box:
        """Clip to the frame. Detectors routinely return boxes that stick out."""
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        return Box(x=x, y=y, w=min(self.w, 1.0 - x), h=min(self.h, 1.0 - y))

    def intersection_area(self, other: Box) -> float:
        dx = min(self.right, other.right) - max(self.x, other.x)
        dy = min(self.bottom, other.bottom) - max(self.y, other.y)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy

    def iou(self, other: Box) -> float:
        inter = self.intersection_area(other)
        if inter == 0.0:
            return 0.0
        return inter / (self.area + other.area - inter)

    def containment_in(self, other: Box) -> float:
        """Fraction of *this* box that falls inside `other`.

        Used to attach a face to the person it belongs to: a face sits almost
        entirely within its person's box, while IoU between the two is tiny
        because the person is so much larger.
        """
        if self.area == 0.0:
            return 0.0
        return self.intersection_area(other) / self.area

    def scaled_to(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Back to pixels, as (x, y, w, h). Only the renderer should call this."""
        return (
            round(self.x * width),
            round(self.y * height),
            round(self.w * width),
            round(self.h * height),
        )


class TimedBox(Box):
    """A box at a point in time."""

    t: float


class Segment(BaseModel):
    """A half-open interval [start, end) during which an object is on screen."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    def overlap(self, start: float, end: float) -> Segment | None:
        lo, hi = max(self.start, start), min(self.end, end)
        return Segment(start=lo, end=hi) if hi > lo else None


class Track(BaseModel):
    """One object followed through the video.

    `parent_id` links a face or plate to the person or vehicle carrying it. It
    is what makes "the man in the black cap -- just his face" work: grounding
    identifies the *person* (the cap is only visible on the person box), and the
    blur has to land on that person's *face*.
    """

    id: str
    kind: ObjectKind
    parent_id: str | None = None
    display_name: str | None = None
    segments: list[Segment] = Field(default_factory=list)

    @property
    def first_seen(self) -> float | None:
        return self.segments[0].start if self.segments else None

    @property
    def last_seen(self) -> float | None:
        return self.segments[-1].end if self.segments else None

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def present_at(self, t: float) -> bool:
        return any(s.contains(t) for s in self.segments)

    def segments_within(self, start: float, end: float) -> list[Segment]:
        """The parts of this track's presence that fall inside [start, end)."""
        return [seg for s in self.segments if (seg := s.overlap(start, end))]

    def gaps_within(self, start: float, end: float) -> list[Segment]:
        """Stretches of [start, end) where this track is absent.

        A gap is ambiguous on its own -- the person may have walked off, or the
        tracker may have lost them behind someone else. Only the user can say
        which, so these surface as a confirmation warning rather than being
        silently filled in.
        """
        gaps: list[Segment] = []
        cursor = start
        for seg in self.segments_within(start, end):
            if seg.start > cursor:
                gaps.append(Segment(start=cursor, end=seg.start))
            cursor = max(cursor, seg.end)
        if cursor < end:
            gaps.append(Segment(start=cursor, end=end))
        return gaps


class TrackBoxes(BaseModel):
    """Per-frame geometry for one track, kept out of timeline responses."""

    track_id: str
    boxes: list[TimedBox] = Field(default_factory=list)

    def at(self, t: float, max_gap: float = 0.5) -> Box | None:
        """Box at time `t`, linearly interpolated between samples.

        Returns None when `t` falls in a gap wider than `max_gap`, so a blur
        never drifts across a stretch where the object was not actually seen.
        """
        if not self.boxes:
            return None

        before: TimedBox | None = None
        after: TimedBox | None = None
        for box in self.boxes:
            if box.t <= t:
                before = box
            else:
                after = box
                break

        if before is not None and before.t == t:
            return Box(x=before.x, y=before.y, w=before.w, h=before.h)
        if before is None or after is None:
            edge = before or after
            assert edge is not None
            return (
                Box(x=edge.x, y=edge.y, w=edge.w, h=edge.h) if abs(edge.t - t) <= max_gap else None
            )
        if after.t - before.t > max_gap:
            return None

        ratio = (t - before.t) / (after.t - before.t)
        return Box(
            x=before.x + (after.x - before.x) * ratio,
            y=before.y + (after.y - before.y) * ratio,
            w=before.w + (after.w - before.w) * ratio,
            h=before.h + (after.h - before.h) * ratio,
        )
