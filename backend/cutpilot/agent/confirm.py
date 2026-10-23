"""Draft + groundings -> an executable EditPlan and the screen that asks about it.

This is where the confirmation rules live:

    ask only where it buys something   per-field status from grounding
    say what only a render would show   three warnings computed from boxes

Warnings are cheap -- interval arithmetic and box intersections at a coarse
sample rate -- and each one replaces a render the user would otherwise have
watched to find the same thing out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from cutpilot.agent.grounder import Candidate, FrameLoader, Grounding, ground_phrase, is_unresolved
from cutpilot.agent.llm import LLMProvider
from cutpilot.editing import reframe as reframe_geometry
from cutpilot.schemas.confirm import (
    Confirmation,
    ConfirmItem,
    ConfirmStatus,
    Overlay,
    PlanWarning,
)
from cutpilot.schemas.plan import (
    EditPlan,
    EditPlanDraft,
    Operation,
    OperationDraft,
    Selector,
    TimeRange,
)
from cutpilot.schemas.session import UIContext
from cutpilot.schemas.track import Box, Track, TrackBoxes
from cutpilot.schemas.video import VideoAsset
from cutpilot.vision.resolve import resolve_selector

STATUS_RANK: dict[ConfirmStatus, int] = {"ok": 0, "confirm": 1, "choose": 2}
OP_LABEL = {"MOSAIC": "모자이크", "BLUR": "블러", "REFRAME": "세로 변환"}
KIND_LABEL = {"face": "얼굴", "person": "사람", "license_plate": "번호판"}

GAP_MIN_SECONDS = 0.3
"""Shorter interior gaps are tracker jitter, already bridged in segments."""

OUT_OF_FRAME_SAMPLE = 0.5
OUT_OF_FRAME_SHARE = 0.5
"""A target outside the crop for more than half its screen time counts as
cropped away -- the blur is spent on pixels the output never shows."""

CANDIDATE_LIMIT = 6


@dataclass
class GroundedPlan:
    plan: EditPlan
    confirmation: Confirmation
    groundings: dict[tuple[int, str, str], Grounding] = field(default_factory=dict)
    """Keyed by (operation index, "include" | "exclude", phrase)."""


def _worst(statuses: list[ConfirmStatus]) -> ConfirmStatus:
    return max(statuses, key=STATUS_RANK.__getitem__, default="ok")


def _dedupe(ids: list[str]) -> list[str]:
    out: list[str] = []
    for track_id in ids:
        if track_id not in out:
            out.append(track_id)
    return out


def _most_present(tracks: list[Track]) -> list[Candidate]:
    people = sorted(
        (t for t in tracks if t.kind == "person"), key=lambda t: t.total_duration, reverse=True
    )
    return [Candidate(track_id=t.id, score=0.0) for t in people[:CANDIDATE_LIMIT]]


# -------------------------------------------------------------- grounding


def ground_draft(
    provider: LLMProvider,
    draft: EditPlanDraft,
    *,
    asset: VideoAsset,
    tracks: list[Track],
    boxes: dict[str, TrackBoxes],
    frame_loader: FrameLoader,
    ui: UIContext | None = None,
    thinking_level: str | None = "minimal",
    grounder: Callable[..., Grounding] = ground_phrase,
) -> GroundedPlan:
    groundings: dict[tuple[int, str, str], Grounding] = {}
    operations: list[Operation] = []
    dropped: list[PlanWarning] = []

    def ground(index: int, side: str, phrase: str, op: OperationDraft) -> Grounding:
        result = grounder(
            provider,
            phrase,
            kind=op.selector.kind,
            time_range=op.time_range,
            duration=asset.duration,
            tracks=tracks,
            boxes=boxes,
            frame_loader=frame_loader,
            ui=ui,
            thinking_level=thinking_level,
        )
        groundings[(index, side, phrase)] = result
        return result

    for index, op in enumerate(draft.operations):
        includes = [ground(index, "include", p, op) for p in op.selector.include_phrases]
        excludes = [ground(index, "exclude", p, op) for p in op.selector.exclude_phrases]

        include_ids = _dedupe([i for g in includes for i in g.ids_for_plan])
        exclude_ids = _dedupe([i for g in excludes for i in g.ids_for_plan])

        if op.op == "REFRAME":
            real = [i for i in include_ids if not is_unresolved(i)]
            if len(real) != 1:
                # No subject named, nothing matched, or several did. A reframe
                # follows exactly one person, so offer the most-present ones.
                pool = [c for g in includes for c in g.candidates] or _most_present(tracks)
                provisional = real[0] if real else pool[0].track_id if pool else None
                key = (index, "include", op.selector.include_phrases[0] if includes else "")
                groundings[key] = Grounding(
                    phrase=key[2],
                    status="choose",
                    chosen=[provisional] if provisional else [],
                    candidates=pool,
                )
                include_ids = [provisional] if provisional else []
            if not include_ids:
                dropped.append(
                    PlanWarning(
                        kind="empty_selection",
                        message="세로 변환의 기준이 될 사람을 영상에서 찾지 못했습니다.",
                    )
                )
                continue
            exclude_ids = []

        operations.append(
            Operation(
                op=op.op,
                selector=Selector(kind=op.selector.kind, include=include_ids, exclude=exclude_ids),
                time_range=op.time_range,
                aspect_ratio=op.aspect_ratio,
                strength=op.strength,
            )
        )

    plan = EditPlan(video_id=asset.id, operations=operations)
    confirmation = build_confirmation(
        plan, groundings, asset=asset, tracks=tracks, boxes=boxes, ui=ui
    )
    confirmation.warnings[:0] = dropped
    return GroundedPlan(plan=plan, confirmation=confirmation, groundings=groundings)


# ------------------------------------------------------------- confirmation


def _selector_label(op: Operation, groundings: dict, index: int) -> str:
    kind = KIND_LABEL.get(op.selector.kind, op.selector.kind)
    phrases_in = [k[2] for k in groundings if k[0] == index and k[1] == "include" and k[2]]
    phrases_out = [k[2] for k in groundings if k[0] == index and k[1] == "exclude"]
    if op.op == "REFRAME":
        return f"기준 인물: {phrases_in[0] if phrases_in else op.selector.include[0]}"
    target = (
        ", ".join(phrases_in)
        if phrases_in
        else (", ".join(op.selector.include) if op.selector.include else "전체")
    )
    label = f"{kind} · {target}"
    if phrases_out or op.selector.exclude:
        label += f" (제외: {', '.join(phrases_out or op.selector.exclude)})"
    return label


def _interior_gaps(track: Track, lo: float, hi: float):
    within = track.segments_within(lo, hi)
    if len(within) < 2:
        return []
    first, last = within[0].start, within[-1].end
    return [g for g in track.gaps_within(first, last) if g.duration >= GAP_MIN_SECONDS]


def _reframe_windows(plan: EditPlan, asset: VideoAsset, tracks, boxes) -> list[Box]:
    op = plan.reframe
    if op is None:
        return []
    subject = resolve_selector(op.selector, tracks)
    if not subject or subject[0].id not in boxes:
        return []
    return reframe_geometry.crop_trajectory(
        boxes[subject[0].id],
        source_w=asset.width,
        source_h=asset.height,
        target_ratio=reframe_geometry.parse_aspect_ratio(op.aspect_ratio or "9:16"),
        duration=asset.duration,
        fps=asset.fps,
    )


def build_confirmation(
    plan: EditPlan,
    groundings: dict[tuple[int, str, str], Grounding],
    *,
    asset: VideoAsset,
    tracks: list[Track],
    boxes: dict[str, TrackBoxes],
    ui: UIContext | None = None,
) -> Confirmation:
    by_id = {t.id: t for t in tracks}
    items: list[ConfirmItem] = []
    warnings: list[PlanWarning] = []

    for index, op in enumerate(plan.operations):
        own = [g for k, g in groundings.items() if k[0] == index]
        status = _worst([g.status for g in own])
        candidates = _dedupe(
            [c.track_id for g in own if g.status == "choose" for c in g.candidates]
        )

        op_label = OP_LABEL[op.op] + (f" {op.aspect_ratio}" if op.aspect_ratio else "")
        items.append(ConfirmItem(operation_index=index, field="op", label=op_label))
        items.append(
            ConfirmItem(
                operation_index=index,
                field="selector",
                label=_selector_label(op, groundings, index),
                status=status,
                candidates=candidates[:CANDIDATE_LIMIT],
            )
        )
        items.append(
            ConfirmItem(
                operation_index=index,
                field="time_range",
                label=op.time_range.describe(asset.duration),
            )
        )

        targets = resolve_selector(op.selector, tracks)
        if not targets:
            warnings.append(
                PlanWarning(
                    kind="empty_selection",
                    operation_index=index,
                    message=(
                        f"'{_selector_label(op, groundings, index)}'에 해당하는 대상을 "
                        "영상에서 찾지 못했습니다."
                    ),
                )
            )
            continue

        lo, hi = op.time_range.resolve(asset.duration)
        if op.selector.include:  # explicit targets only; whole-kind gaps are just exits
            for track in targets:
                for gap in _interior_gaps(track, lo, hi):
                    risk = " 가려진 거라면 그 구간은 처리되지 않습니다." if op.is_privacy_op else ""
                    warnings.append(
                        PlanWarning(
                            kind="track_gap",
                            operation_index=index,
                            track_id=track.id,
                            time_range=TimeRange(start=gap.start, end=gap.end),
                            message=(
                                f"{track.id}이(가) {gap.start:.1f}~{gap.end:.1f}s 동안 "
                                f"보이지 않습니다. 화면을 벗어난 건가요, 가려진 건가요?{risk}"
                            ),
                        )
                    )

    windows = _reframe_windows(plan, asset, tracks, boxes)
    if windows:
        for index, op in enumerate(plan.operations):
            if not op.is_privacy_op:
                continue
            lo, hi = op.time_range.resolve(asset.duration)
            cropped = 0
            for track in resolve_selector(op.selector, tracks):
                seen = outside = 0
                t = lo
                while t < hi:
                    box = boxes[track.id].at(t) if track.id in boxes else None
                    if box is not None and track.present_at(t):
                        seen += 1
                        frame = min(int(t * asset.fps), len(windows) - 1)
                        if windows[frame].intersection_area(box) == 0.0:
                            outside += 1
                    t += OUT_OF_FRAME_SAMPLE
                if seen and outside / seen > OUT_OF_FRAME_SHARE:
                    cropped += 1
            if cropped:
                warnings.append(
                    PlanWarning(
                        kind="out_of_frame",
                        operation_index=index,
                        message=(
                            f"{OP_LABEL[op.op]} 대상 중 {cropped}개는 세로 변환 후 대부분 화면 "
                            "밖이라 결과에 거의 나타나지 않습니다."
                        ),
                    )
                )

    keyframe = next((g.t for g in groundings.values() if g.t is not None), None)
    if keyframe is None:
        keyframe = ui.playhead if ui and ui.playhead is not None else 0.0

    return Confirmation(
        plan=plan,
        keyframe_t=keyframe,
        overlays=_overlays(plan, groundings, keyframe, tracks, boxes, by_id),
        items=items,
        warnings=warnings,
    )


def _overlays(plan, groundings, t, tracks, boxes, by_id) -> list[Overlay]:
    out: list[Overlay] = []

    def add(track_id: str, role, index: int, label: str | None = None) -> None:
        source = boxes.get(track_id)
        track = by_id.get(track_id)
        box = source.at(t) if source and track and track.present_at(t) else None
        if box is not None:
            out.append(
                Overlay(track_id=track_id, box=box, role=role, label=label, operation_index=index)
            )

    for index, op in enumerate(plan.operations):
        if op.op == "REFRAME":
            for track in resolve_selector(op.selector, tracks):
                add(track.id, "reframe_subject", index, "세로 변환 기준")
            continue
        for track in resolve_selector(op.selector, tracks):
            add(track.id, "target", index)
        for track_id in op.selector.exclude:
            if not is_unresolved(track_id):
                add(track_id, "excluded", index, "제외")

    for (index, _side, _phrase), g in groundings.items():
        if g.status == "choose":
            for c in g.candidates[:CANDIDATE_LIMIT]:
                add(c.track_id, "candidate", index)
    return out
