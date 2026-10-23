"""Call 2 of 2: a phrase and one frame -> which tracks it means.

Gemini picks, YOLO measures. Gemini's boxes are approximate, so they are only
used to decide *which* tracked object a phrase refers to; every coordinate the
edit uses comes from the tracker. The match is IoU between the two.

The match also decides how sure we are, which is where the confirmation step's
three states come from:

    ok       the phrase was a track id -- a click, or a target carried over
             from an earlier turn. No model call, nothing to ask.
    confirm  one track matched clearly. Show it and ask.
    choose   several matched, or two overlap too closely to separate, or
             nothing matched. Do not guess; let the user pick.

One trap is closed here by construction. A phrase that grounds to nothing must
not leave an empty include list, because an empty include means *everyone*:
"just the man in the black cap" would silently become "all faces". Unresolved
phrases are carried as an UNRESOLVED marker id that selects nobody, so the
failure surfaces as an empty selection the user can see.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from cutpilot.agent.llm import LLMProvider, generate_structured
from cutpilot.agent.planner import track_id_of
from cutpilot.agent.prompts import GROUNDING_PROMPT
from cutpilot.schemas.confirm import ConfirmStatus
from cutpilot.schemas.plan import TimeRange
from cutpilot.schemas.session import UIContext
from cutpilot.schemas.track import Box, ObjectKind, Track, TrackBoxes

MIN_IOU = 0.3
"""Below this a Gemini box is not considered to be that track at all."""

AMBIGUITY_MARGIN = 0.15
"""If the runner-up track scores within this of the best, the box sits on two
people and the phrase cannot separate them."""

UNRESOLVED_PREFIX = "?"
SCAN_STEP = 0.5
MAX_FRAME_SIDE = 1280

FrameLoader = Callable[[float], bytes]


# ----------------------------------------------------------- model output


class DetectedObject(BaseModel):
    box_2d: list[int] = Field(min_length=4, max_length=4)
    """[ymin, xmin, ymax, xmax], 0-1000 -- Gemini's native detection format."""
    label: str = ""


class Detections(BaseModel):
    objects: list[DetectedObject] = Field(default_factory=list)


def to_box(detected: DetectedObject) -> Box:
    ymin, xmin, ymax, xmax = (max(0, min(1000, v)) / 1000 for v in detected.box_2d)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return Box(x=xmin, y=ymin, w=xmax - xmin, h=ymax - ymin)


# ----------------------------------------------------------------- result


@dataclass
class Candidate:
    track_id: str
    score: float


@dataclass
class Grounding:
    phrase: str
    status: ConfirmStatus
    t: float | None = None
    kind: ObjectKind = "person"
    chosen: list[str] = field(default_factory=list)
    """Provisional answer. Under "choose" this is a best guess the user still
    has to accept; it is never acted on without approval."""
    candidates: list[Candidate] = field(default_factory=list)
    detected: list[Box] = field(default_factory=list)

    @property
    def ids_for_plan(self) -> list[str]:
        """What goes into the selector. Never empty -- see module docstring."""
        return self.chosen or [unresolved_id(self.phrase)]


def unresolved_id(phrase: str) -> str:
    return f"{UNRESOLVED_PREFIX}{phrase}"


def is_unresolved(track_id: str) -> bool:
    return track_id.startswith(UNRESOLVED_PREFIX)


# ---------------------------------------------------------------- keyframe


def _present(tracks: list[Track], boxes: dict[str, TrackBoxes], kind: ObjectKind, t: float):
    out = []
    for track in tracks:
        if track.kind != kind or not track.present_at(t):
            continue
        source = boxes.get(track.id)
        box = source.at(t) if source else None
        if box is not None:
            out.append((track, box))
    return out


def pick_keyframe(
    time_range: TimeRange,
    *,
    duration: float,
    kind: ObjectKind,
    tracks: list[Track],
    boxes: dict[str, TrackBoxes],
    ui: UIContext | None = None,
) -> float:
    """Where to look. The playhead if it falls in the range -- that is the frame
    the user was looking at when they spoke -- else the start of the range,
    moved forward to the busiest moment if nobody is on screen there."""
    start, end = time_range.resolve(duration)
    t = start
    if ui and ui.playhead is not None and start <= ui.playhead < end:
        t = ui.playhead
    if _present(tracks, boxes, kind, t):
        return t

    best_t, best_n = t, 0
    probe = start
    while probe < end:
        n = len(_present(tracks, boxes, kind, probe))
        if n > best_n:
            best_t, best_n = probe, n
        probe += SCAN_STEP
    return best_t


# ---------------------------------------------------------------- matching


def match(detected: list[Box], present: list[tuple[Track, Box]]) -> tuple[list[Candidate], bool]:
    """Best track per detected box, plus whether any box sat on two tracks."""
    best: dict[str, float] = {}
    ambiguous = False
    for box in detected:
        scored = sorted(
            ((track.id, box.iou(track_box)) for track, track_box in present),
            key=lambda item: item[1],
            reverse=True,
        )
        if not scored or scored[0][1] < MIN_IOU:
            continue
        top_id, top_score = scored[0]
        if len(scored) > 1 and scored[1][1] >= MIN_IOU:
            if top_score - scored[1][1] < AMBIGUITY_MARGIN:
                ambiguous = True
                best[scored[1][0]] = max(best.get(scored[1][0], 0.0), scored[1][1])
        best[top_id] = max(best.get(top_id, 0.0), top_score)

    candidates = [Candidate(track_id=k, score=v) for k, v in best.items()]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates, ambiguous


def ground_phrase(
    provider: LLMProvider,
    phrase: str,
    *,
    kind: ObjectKind,
    time_range: TimeRange,
    duration: float,
    tracks: list[Track],
    boxes: dict[str, TrackBoxes],
    frame_loader: FrameLoader,
    ui: UIContext | None = None,
    thinking_level: str | None = "minimal",
) -> Grounding:
    direct = track_id_of(phrase)
    if direct is not None:
        known = any(t.id == direct for t in tracks)
        return Grounding(
            phrase=phrase,
            status="ok" if known else "choose",
            chosen=[direct] if known else [],
        )

    # Appearance lives on the whole object: a cap or a coat is visible on the
    # person box, never on the face box. Faces are reached afterwards through
    # parent links; plates are matched as plates.
    target_kind: ObjectKind = "license_plate" if kind == "license_plate" else "person"
    t = pick_keyframe(
        time_range, duration=duration, kind=target_kind, tracks=tracks, boxes=boxes, ui=ui
    )
    present = _present(tracks, boxes, target_kind, t)
    all_present = [Candidate(track_id=track.id, score=0.0) for track, _ in present]

    result, _ = generate_structured(
        provider,
        Detections,
        GROUNDING_PROMPT.format(phrase=phrase),
        images=(frame_loader(t),),
        thinking_level=thinking_level,
    )
    detected = [to_box(obj) for obj in result.objects]
    candidates, ambiguous = match(detected, present)

    if not candidates:
        # Nothing fits -- offer everyone on screen rather than inventing a match.
        return Grounding(
            phrase=phrase, status="choose", t=t, kind=target_kind,
            candidates=all_present, detected=detected,
        )  # fmt: skip
    status: ConfirmStatus = "choose" if (ambiguous or len(candidates) > 1) else "confirm"
    chosen = [candidates[0].track_id] if ambiguous else [c.track_id for c in candidates]
    return Grounding(
        phrase=phrase, status=status, t=t, kind=target_kind,
        chosen=chosen, candidates=candidates, detected=detected,
    )  # fmt: skip


# ------------------------------------------------------------------ frames


def video_frame_loader(path: Path, fps: float, *, max_side: int = MAX_FRAME_SIDE) -> FrameLoader:
    """Read one frame at time t as JPEG, downscaled for the request."""

    def load(t: float) -> bytes:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"cannot read frame at {t:.2f}s from {path}")
        h, w = frame.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return encoded.tobytes()

    return load
