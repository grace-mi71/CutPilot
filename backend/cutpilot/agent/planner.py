"""Call 1 of 2: a sentence, the conversation and the screen -> EditPlanDraft.

Text only, no image. The planner sees what is on the timeline as a list of
tracks and when each is present, which is enough to resolve references like
"whoever is there from the start" without looking at a frame. Pointing at a
specific person by appearance is left as a phrase for the grounder, the one
step that actually needs pixels.

What comes back is validated for shape by the provider, then normalised here
for meaning. The two are different checks: the first live probe returned a
valid plan that tried to ground "the other people" as if it were a person.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from cutpilot.agent.llm import LLMProvider, LLMResponse, generate_structured
from cutpilot.agent.prompts import PLANNER_SYSTEM
from cutpilot.schemas.plan import EditPlan, EditPlanDraft, OperationDraft, TimeRange
from cutpilot.schemas.session import UIContext
from cutpilot.schemas.track import Track

TRACK_ID = re.compile(r"^@?((?:person|face|license_plate)_\d+)$")

# Words that pick out "everyone" rather than someone. A phrase made only of
# these is a complement, not a target: "the other people" with an exclusion is
# the whole kind minus the exclusion, and grounding it would find nobody or,
# worse, someone arbitrary.
COMPLEMENT_TOKENS = frozenset(
    {
        "다른", "나머지", "모든", "모두", "전부", "전체", "다", "다들",
        "사람", "사람들", "인물", "인물들", "얼굴", "얼굴들", "번호판", "번호판들",
        "everyone", "everybody", "others", "other", "the", "rest", "all",
        "people", "person", "faces", "face",
    }
)  # fmt: skip
PARTICLES = ("은", "는", "이", "가", "을", "를", "의", "도")

HISTORY_TURNS = 6


@dataclass
class Turn:
    role: str  # "user" | "agent"
    text: str


@dataclass
class PlannerResult:
    draft: EditPlanDraft
    notes: list[str] = field(default_factory=list)
    """Corrections made during normalisation, worth showing the user."""
    response: LLMResponse | None = None


# --------------------------------------------------------------- normalising


def track_id_of(phrase: str) -> str | None:
    """Return the track id a phrase names directly ("@person_03"), else None."""
    match = TRACK_ID.match(phrase.strip())
    return match.group(1) if match else None


def _strip_particle(token: str) -> str:
    for particle in PARTICLES:
        if len(token) > len(particle) and token.endswith(particle):
            stem = token[: -len(particle)]
            if stem in COMPLEMENT_TOKENS:
                return stem
    return token


def is_complement(phrase: str) -> bool:
    tokens = [_strip_particle(t) for t in phrase.lower().split()]
    return bool(tokens) and all(t in COMPLEMENT_TOKENS for t in tokens)


def _clean_phrases(phrases: list[str]) -> list[str]:
    seen: list[str] = []
    for phrase in phrases:
        # A leading "?" is the grounder's unresolved marker; the phrase is what
        # it stood for, never part of a name.
        text = phrase.strip().lstrip("?").strip()
        text = track_id_of(text) or text
        if text and text not in seen:
            seen.append(text)
    return seen


def _clamp_time(range_: TimeRange, duration: float) -> TimeRange:
    def clamp(value: float | None) -> float | None:
        return None if value is None else min(max(value, 0.0), duration)

    start, end = clamp(range_.start), clamp(range_.end)
    if start is not None and end is not None and start > end:
        start, end = end, start
    return TimeRange(start=start, end=end)


def normalize_draft(draft: EditPlanDraft, duration: float) -> tuple[EditPlanDraft, list[str]]:
    """Enforce in code the conventions the prompt asks for."""
    notes: list[str] = []
    operations: list[OperationDraft] = []

    for op in draft.operations:
        selector = op.selector.model_copy()
        include = _clean_phrases(selector.include_phrases)
        exclude = _clean_phrases(selector.exclude_phrases)

        complements = [p for p in include if is_complement(p)]
        if complements:
            include = [p for p in include if p not in complements]
            notes.append(
                f"'{', '.join(complements)}'은(는) 특정 대상이 아니라 전체를 뜻해 "
                "대상 목록에서 뺐습니다."
            )
        exclude = [p for p in exclude if not is_complement(p)]

        kind = selector.kind
        aspect_ratio, strength = op.aspect_ratio, op.strength
        if op.op == "REFRAME":
            kind = "person"
            aspect_ratio = aspect_ratio or "9:16"
            strength = None
            exclude = []
        else:
            aspect_ratio = None

        operations.append(
            OperationDraft(
                op=op.op,
                selector=selector.model_copy(
                    update={"kind": kind, "include_phrases": include, "exclude_phrases": exclude}
                ),
                time_range=_clamp_time(op.time_range, duration),
                aspect_ratio=aspect_ratio,
                strength=strength,
            )
        )

    reframes = [o for o in operations if o.op == "REFRAME"]
    if len(reframes) > 1:
        notes.append("화면 비율 변환은 하나만 적용할 수 있어 첫 번째만 남겼습니다.")
        first = reframes[0]
        operations = [o for o in operations if o.op != "REFRAME" or o is first]

    return draft.model_copy(update={"operations": operations}), notes


# ----------------------------------------------------------------- context


def _fmt(t: float) -> str:
    return f"{t:.1f}"


def track_summary(tracks: list[Track]) -> str:
    lines = []
    for track in sorted(tracks, key=lambda t: (t.kind, t.id)):
        spans = ", ".join(f"{_fmt(s.start)}–{_fmt(s.end)}s" for s in track.segments)
        name = f"  이름: {track.display_name}" if track.display_name else ""
        lines.append(f"- {track.id} ({track.kind})  등장 {spans}{name}")
    return "\n".join(lines) if lines else "(탐지된 대상 없음)"


def ui_summary(ui: UIContext | None, duration: float) -> str:
    lines = [f"영상 길이: {_fmt(duration)}s"]
    if ui and ui.playhead is not None:
        lines.append(f"재생 위치: {_fmt(ui.playhead)}s")
    if ui and ui.selection is not None:
        start, end = ui.selection.resolve(duration)
        lines.append(f"선택 구간: {_fmt(start)}s ~ {_fmt(end)}s")
    else:
        lines.append("선택 구간: 없음")
    return "\n".join(lines)


def build_prompt(
    utterance: str,
    *,
    tracks: list[Track],
    duration: float,
    ui: UIContext | None = None,
    current_plan: EditPlan | None = None,
    history: list[Turn] | None = None,
) -> str:
    sections = [
        "## UI 상태\n" + ui_summary(ui, duration),
        "## 영상 속 대상\n" + track_summary(tracks),
    ]
    if current_plan is not None and current_plan.operations:
        payload = [op.model_dump(mode="json") for op in current_plan.operations]
        sections.append(
            "## 현재 계획 (수정 요청이면 이걸 바탕으로)\n"
            + json.dumps(payload, ensure_ascii=False, indent=1)
        )
    if history:
        recent = history[-HISTORY_TURNS:]
        sections.append("## 이전 대화\n" + "\n".join(f"{t.role}: {t.text}" for t in recent))
    sections.append(f"## 사용자 요청\n{utterance}")
    return "\n\n".join(sections)


# ------------------------------------------------------------------- entry


def parse_intent(
    provider: LLMProvider,
    utterance: str,
    *,
    tracks: list[Track],
    duration: float,
    ui: UIContext | None = None,
    current_plan: EditPlan | None = None,
    history: list[Turn] | None = None,
    thinking_level: str | None = "low",
) -> PlannerResult:
    prompt = build_prompt(
        utterance,
        tracks=tracks,
        duration=duration,
        ui=ui,
        current_plan=current_plan,
        history=history,
    )
    draft, response = generate_structured(
        provider,
        EditPlanDraft,
        prompt,
        system=PLANNER_SYSTEM,
        thinking_level=thinking_level,
    )
    normalized, notes = normalize_draft(draft, duration)
    return PlannerResult(draft=normalized, notes=notes, response=response)
