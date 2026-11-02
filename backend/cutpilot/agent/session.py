"""The negotiation loop: plan once, answer the agent's questions, approve the spec.

The user hands over a plan -- one sentence or several numbered steps. The agent
works through it and stops wherever it cannot go on alone, one question at a
time (agent/questions.py). Once nothing is left to ask, it writes out the spec
it will run, and only an approved spec is rendered.

Everything before this module turns one message into one confirmation. This is
the state that survives between turns and the ways the user answers:

    say(text)                 the plan, a spoken answer, or a correction
    answer(question, option)  a one-click answer ("맞아요", "그 구간도 처리")
    choose(op, ids)           answer a "which one?" by clicking a candidate
    toggle(op, track_id)      click a box on the preview to add or drop it
    set_time_range(op, ...)   drag the range on the timeline
    remove(op)                drop an operation

Only `say` calls a model. Every click edits the plan directly, so correcting
the agent never costs a round trip and never risks the model reinterpreting
what was already settled.

Two rules are enforced here rather than trusted to the UI:

1. Nothing renders unapproved, and approval is refused while any question is
   open: a choice, a phrase that found nobody, an operation that selects
   nothing, or a check the agent raised and the user has not answered.
2. An open choice stays open across turns. A provisional answer is never shown
   to the planner as a settled id -- "only up to 15 seconds" must not quietly
   confirm a guess the user never accepted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from cutpilot.agent.confirm import GroundedPlan, build_confirmation, ground_draft
from cutpilot.agent.grounder import (
    Candidate,
    FrameLoader,
    Grounding,
    ground_phrase,
    is_unresolved,
    present_at,
    unresolved_id,
)
from cutpilot.agent.llm import LLMError, LLMProvider, LLMQuotaExhausted, LLMRequest, LLMResponse
from cutpilot.agent.llm.budget import LLMBudgetExceeded
from cutpilot.agent.planner import Turn, parse_intent, track_id_of
from cutpilot.agent.questions import OpenQuestion, build_spec, open_questions, shift_acknowledged
from cutpilot.schemas.confirm import Confirmation
from cutpilot.schemas.plan import EditPlan, Operation, TimeRange
from cutpilot.schemas.question import Question, Spec
from cutpilot.schemas.session import UIContext
from cutpilot.schemas.track import Segment, TimedBox, Track, TrackBoxes
from cutpilot.schemas.video import VideoAsset

ReplyStatus = Literal["plan", "clarify", "error"]
EventKind = Literal[
    "say", "answer", "choose", "toggle", "time_range", "remove", "approve", "render"
]

GroundingKey = tuple[int, str, str]


class ApprovalBlocked(Exception):
    """Approval was refused. `reasons` is what the user still has to settle."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class Reply:
    status: ReplyStatus
    text: str
    confirmation: Confirmation | None = None
    notes: list[str] = field(default_factory=list)
    question: Question | None = None
    """What the agent is waiting on now. None once everything is settled."""
    remaining: int = 0
    """Open questions, this one included."""
    spec: Spec | None = None
    """Set once nothing is left to ask: what approval will render."""
    blockers: list[str] = field(default_factory=list)

    @property
    def can_approve(self) -> bool:
        return self.confirmation is not None and not self.blockers


@dataclass
class SessionEvent:
    """One step of the negotiation, kept for the week-11 comparison: how many
    corrections and model calls it took to reach an approved plan."""

    kind: EventKind
    detail: str = ""
    llm_calls: int = 0


class _CountingProvider:
    """Counts the calls that actually reached a model (cache hits excluded)."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        response = self.inner.generate(request)
        if not response.cached:
            self.calls += 1
        return response


YES_WORDS = frozenset(
    {"네", "예", "응", "어", "웅", "맞아", "맞아요", "맞습니다", "맞음", "그래", "그래요", "좋아",
     "좋아요", "괜찮아", "괜찮아요", "그렇게", "그렇게해", "그렇게해줘", "해줘", "ㅇㅇ", "ok",
     "okay", "yes"}
)  # fmt: skip
NO_WORDS = frozenset({"아니", "아니요", "아니오", "아뇨", "아냐", "아니야", "ㄴㄴ", "no"})


def quick_option(utterance: str, question: Question) -> str | None:
    """A bare yes or no to a yes/no question, answered without the model.

    Only a reply that is nothing *but* yes or no counts: "아니 5초부터" is a
    correction and goes to the planner.
    """
    if question.blocking or not question.options:
        return None
    word = "".join(ch for ch in utterance.lower() if ch.isalnum())
    if word in YES_WORDS:
        return question.options[0].id
    if word in NO_WORDS and len(question.options) > 1:
        return question.options[1].id
    return None


# ----------------------------------------------------------------- blockers


def approval_blockers(confirmation: Confirmation) -> list[str]:
    plan = confirmation.plan
    if not plan.operations:
        return ["승인할 편집 작업이 없습니다."]

    reasons = [
        f"{item.operation_index + 1}번 작업: 대상을 골라 주세요 "
        f"(후보: {', '.join(item.candidates) or '없음'})."
        for item in confirmation.blocking_items
    ]
    for index, op in enumerate(plan.operations):
        for track_id in (*op.selector.include, *op.selector.exclude):
            if is_unresolved(track_id):
                reasons.append(f"{index + 1}번 작업: '{track_id[1:]}'을(를) 찾지 못했습니다.")
    reasons += [
        w.message
        for w in confirmation.warnings
        if w.kind == "empty_selection" and w.operation_index is not None
    ]
    return list(dict.fromkeys(reasons))


# ---------------------------------------------------------------- describing


def describe(confirmation: Confirmation) -> list[str]:
    """The plan as the agent understood it, one line per step."""
    lines = ["이렇게 이해했습니다:"]
    for index in range(len(confirmation.plan.operations)):
        items = {item.field: item for item in confirmation.items_for(index)}
        parts = [items[f].label for f in ("op", "selector", "time_range") if f in items]
        lines.append(f"{index + 1}. " + " · ".join(parts))
    return lines


def narrate(
    confirmation: Confirmation,
    question: Question | None,
    remaining: int,
    spec: Spec | None,
    *,
    notes: list[str] | None = None,
    lead: str | None = None,
) -> str:
    """The agent's turn as chat text: where it got to, then what it needs.

    `lead` replaces the plan summary after an answer -- following "맞아요" the
    user needs the next question, not the whole plan again.
    """
    lines = [lead] if lead else describe(confirmation)
    lines += [f"참고: {n}" for n in notes or []]
    if question is not None:
        more = f" (남은 확인 {remaining}개)" if remaining > 1 else ""
        lines.append(f"진행하다가 확인할 것이 있습니다{more}.")
        lines.append(f"❓ {question.text}")
    elif spec is not None:
        lines.append(spec.text)
    return "\n".join(lines)


def one_line(confirmation: Confirmation) -> str:
    """The same plan in one line, for the conversation history the planner sees."""
    ops = []
    for index in range(len(confirmation.plan.operations)):
        ops.append(" · ".join(i.label for i in confirmation.items_for(index)))
    return "계획: " + (" / ".join(ops) if ops else "없음")


# ------------------------------------------------------------------ session


class EditSession:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        asset: VideoAsset,
        tracks: list[Track],
        boxes: dict[str, TrackBoxes],
        frame_loader: FrameLoader,
        planner_thinking: str | None = "low",
        grounding_thinking: str | None = "minimal",
    ) -> None:
        self.provider = _CountingProvider(provider)
        self.asset = asset
        # Copied: approval names tracks, and the caller's list is not ours to edit.
        self.tracks = [t.model_copy(deep=True) for t in tracks]
        # Shallow copy: bridging a gap replaces one track's boxes, never edits them.
        self.boxes = dict(boxes)
        self.frame_loader = frame_loader
        self.planner_thinking = planner_thinking
        self.grounding_thinking = grounding_thinking

        self.ui = UIContext()
        self.history: list[Turn] = []
        self.current: GroundedPlan | None = None
        self.approved: EditPlan | None = None
        self.events: list[SessionEvent] = []
        self.acknowledged: set[tuple] = set()
        """Non-blocking questions already answered -- never asked twice."""
        self.decisions: list[str] = []
        """Every answer, in words, for the spec."""

    # ------------------------------------------------------------ properties

    @property
    def plan(self) -> EditPlan | None:
        return self.current.plan if self.current else None

    @property
    def confirmation(self) -> Confirmation | None:
        return self.current.confirmation if self.current else None

    @property
    def track_ids(self) -> set[str]:
        return {t.id for t in self.tracks}

    def open_questions(self) -> list[OpenQuestion]:
        return open_questions(self.current, self.acknowledged)

    @property
    def question(self) -> Question | None:
        """What the agent is waiting on, if anything."""
        found = self.open_questions()
        return found[0].question if found else None

    def blockers(self) -> list[str]:
        if self.current is None:
            return ["승인할 편집 작업이 없습니다."]
        reasons = approval_blockers(self.current.confirmation)
        unanswered = [q for q in self.open_questions() if not q.question.blocking]
        if unanswered:
            reasons.append(f"확인할 질문 {len(unanswered)}개에 먼저 답해 주세요.")
        return reasons

    @property
    def spec(self) -> Spec | None:
        if self.current is None or not self.current.plan.operations or self.open_questions():
            return None
        return build_spec(
            self.current, tracks=self.tracks, duration=self.asset.duration,
            decisions=self.decisions,
        )  # fmt: skip

    # ------------------------------------------------------------ speaking

    def say(self, utterance: str, ui: UIContext | None = None) -> Reply:
        """A message from the chat box: the plan, an answer, or a correction."""
        if ui is not None:
            self.ui = ui
        waiting = self.question
        if waiting is not None and (option := quick_option(utterance, waiting)) is not None:
            return self.answer(waiting.id, option, said=utterance)

        before = self.provider.calls
        try:
            planned = parse_intent(
                self.provider,
                utterance,
                tracks=self.tracks,
                duration=self.asset.duration,
                ui=self.ui,
                current_plan=self._plan_for_planner(),
                history=self.history,
                thinking_level=self.planner_thinking,
            )
            draft = planned.draft
            if draft.clarification and not draft.operations:
                reply = Reply(status="clarify", text=draft.clarification, notes=planned.notes)
            else:
                grounded = ground_draft(
                    self.provider,
                    draft,
                    asset=self.asset,
                    tracks=self.tracks,
                    boxes=self.boxes,
                    frame_loader=self.frame_loader,
                    ui=self.ui,
                    thinking_level=self.grounding_thinking,
                    grounder=self._remembering_grounder(),
                )
                self._replace(grounded)
                reply = self._reply(notes=planned.notes)
        except LLMError as exc:
            # The plan on screen is untouched; the user can retry or click.
            self._log("say", f"error: {utterance}", before)
            return self._reply(status="error", lead=_error_text(exc))

        self.history.append(Turn("user", utterance))
        self.history.append(Turn("agent", self._agent_turn(reply)))
        self._log("say", utterance, before)
        return reply

    # ------------------------------------------------------------- answering

    def answer(self, question_id: str, option: str, *, said: str | None = None) -> Reply:
        """A one-click answer to a question the agent asked."""
        found = next((q for q in self.open_questions() if q.question.id == question_id), None)
        if found is None:
            raise IndexError("이미 답했거나 더 이상 해당하지 않는 질문입니다.")
        question = found.question
        if option not in {o.id for o in question.options}:
            raise ValueError(f"이 질문에 없는 답입니다: {option}")
        assert self.current is not None
        index = question.operation_index

        if option == "remove":
            assert index is not None
            self.decisions.append(f"{index + 1}번 작업은 빼기로 했습니다.")
            return self.remove(index)

        if question.kind == "confirm":
            key = found.key
            grounding = self.current.groundings[key]
            ids = ", ".join(grounding.chosen)
            if option == "yes":
                self.current.groundings[key] = replace(grounding, status="ok")
                self.decisions.append(f"'{grounding.phrase}' = {ids} (확인)")
            else:
                self._reopen(key)
                self.decisions.append(f"'{grounding.phrase}'은(는) {ids}이(가) 아님")
        elif question.kind == "track_gap":
            _, track_id, start, end = found.key
            span = f"{track_id} {start:.1f}~{end:.1f}s"
            if option == "fill":
                self.bridge(track_id, start, end)
                self.decisions.append(f"{span}: 가려진 구간도 이어서 처리")
            else:
                self.acknowledged.add(found.ack)
                self.decisions.append(f"{span}: 화면 밖이라 처리하지 않음")
        else:  # out_of_frame keep, dropped-operation ok
            self.acknowledged.add(found.ack)
            if question.kind == "out_of_frame":
                assert index is not None
                self.decisions.append(f"{index + 1}번 작업: 세로 변환 후 잘려도 유지")

        label = next(o.label for o in question.options if o.id == option)
        return self._after_click("answer", said or label, lead=f"알겠습니다: {label}.")

    def _reopen(self, key: tuple) -> None:
        """'No, not that one' -- the guess becomes a choice among the others."""
        assert self.current is not None
        g = self.current.groundings[key]
        rejected = set(g.chosen)
        on_screen = present_at(self.tracks, self.boxes, g.kind, g.t) if g.t is not None else []
        pool = [c for c in g.candidates if c.track_id not in rejected]
        seen = {c.track_id for c in pool}
        pool += [
            Candidate(track_id=t.id, score=0.0)
            for t, _ in on_screen
            if t.id not in rejected and t.id not in seen
        ]
        self.current.groundings[key] = replace(g, status="choose", chosen=[], candidates=pool)
        index, side = key[0], key[1]
        op = self._operation(index)
        ids = [i for i in getattr(op.selector, side) if i not in rejected]
        self._update_selector(index, **{side: [*ids, unresolved_id(g.phrase)]})

    def bridge(self, track_id: str, start: float, end: float) -> None:
        """Cover a gap by carrying the box across it.

        The tracker lost someone behind another person; the user says so. The
        box is interpolated from the last sighting before to the first after,
        and the track counts as present throughout.
        """
        track = next((t for t in self.tracks if t.id == track_id), None)
        source = self.boxes.get(track_id)
        if track is None or source is None:
            raise ValueError(f"영상에 없는 대상입니다: {track_id}")
        before = [b for b in source.boxes if b.t <= start + 1e-6]
        after = [b for b in source.boxes if b.t >= end - 1e-6]
        if not before or not after:
            raise ValueError("구간 앞뒤에 위치가 없어 이어 붙일 수 없습니다.")
        a, b = before[-1], after[0]
        step = 1.0 / self.asset.fps
        filled: list[TimedBox] = []
        t = a.t + step
        while t < b.t - 1e-6:
            r = (t - a.t) / (b.t - a.t)
            filled.append(
                TimedBox(
                    t=round(t, 4),
                    x=a.x + (b.x - a.x) * r,
                    y=a.y + (b.y - a.y) * r,
                    w=a.w + (b.w - a.w) * r,
                    h=a.h + (b.h - a.h) * r,
                )
            )
            t += step
        self.boxes[track_id] = TrackBoxes(
            track_id=track_id, boxes=sorted([*source.boxes, *filled], key=lambda x: x.t)
        )
        segments = sorted([*track.segments, Segment(start=a.t, end=b.t)], key=lambda x: x.start)
        merged: list[Segment] = []
        for seg in segments:
            if merged and seg.start <= merged[-1].end + 1e-6:
                merged[-1] = Segment(start=merged[-1].start, end=max(merged[-1].end, seg.end))
            else:
                merged.append(seg)
        track.segments = merged

    # ------------------------------------------------------------- clicking

    def choose(self, operation_index: int, track_ids: list[str], *, side: str = "include") -> Reply:
        """Answer a "which one?" -- the user picked from the candidates."""
        op = self._operation(operation_index)
        self._check_known(track_ids)
        if op.op == "REFRAME" and len(track_ids) != 1:
            raise ValueError("세로 변환은 한 사람만 따라갈 수 있습니다.")
        if not track_ids:
            raise ValueError("한 명 이상 골라야 합니다.")

        assert self.current is not None
        pending = self._pending(operation_index, side)
        if not pending:
            raise ValueError(f"{operation_index + 1}번 작업에는 고를 항목이 없습니다.")

        stale = {i for key in pending for i in self.current.groundings[key].ids_for_plan}
        phrases = [key[2] for key in pending if key[2]]
        decision = (
            f"'{', '.join(phrases)}' → {', '.join(track_ids)}"
            if phrases
            else f"{operation_index + 1}번 작업 기준 인물 → {', '.join(track_ids)}"
        )
        self.decisions.append(decision)
        for n, key in enumerate(pending):
            # The pick answers the first open phrase; any others it settles too.
            chosen = list(track_ids) if n == 0 else []
            self.current.groundings[key] = replace(
                self.current.groundings[key], status="ok", chosen=chosen
            )

        ids = [i for i in getattr(op.selector, side) if i not in stale]
        ids += [i for i in track_ids if i not in ids]
        self._update_selector(operation_index, **{side: ids})
        return self._after_click(
            "choose",
            f"{operation_index + 1}: {', '.join(track_ids)}",
            lead=f"알겠습니다: {decision}.",
        )

    def toggle(self, operation_index: int, track_id: str) -> Reply:
        """A click on a box in the preview: add that object, or take it out."""
        op = self._operation(operation_index)
        self._check_known([track_id])

        if op.op == "REFRAME" or self._pending(operation_index, "include"):
            # Pointing at someone while a choice is open answers it.
            return self.choose(operation_index, [track_id])

        include, exclude = list(op.selector.include), list(op.selector.exclude)
        if not include:  # whole kind: the click spares or un-spares one
            if track_id in exclude:
                exclude.remove(track_id)
            else:
                exclude.append(track_id)
        elif track_id in include:
            if len(include) == 1:
                # An empty include means *everyone*. Dropping the last target
                # must not widen the edit to the whole kind.
                raise ValueError("마지막 대상은 해제할 수 없습니다. 작업을 삭제해 주세요.")
            include.remove(track_id)
        else:
            include.append(track_id)
            exclude = [i for i in exclude if i != track_id]

        self._update_selector(operation_index, include=include, exclude=exclude)
        return self._after_click("toggle", f"{operation_index + 1}: {track_id}")

    def set_time_range(self, operation_index: int, start: float | None, end: float | None) -> Reply:
        self._operation(operation_index)
        duration = self.asset.duration
        lo = None if start is None else min(max(start, 0.0), duration)
        hi = None if end is None else min(max(end, 0.0), duration)
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("시작이 끝보다 앞서야 합니다.")
        self._update_operation(operation_index, time_range=TimeRange(start=lo, end=hi))
        return self._after_click(
            "time_range", f"{operation_index + 1}: {TimeRange(start=lo, end=hi).describe(duration)}"
        )

    def remove(self, operation_index: int) -> Reply:
        self._operation(operation_index)
        assert self.current is not None
        operations = [
            op for i, op in enumerate(self.current.plan.operations) if i != operation_index
        ]
        groundings = {
            (i if i < operation_index else i - 1, side, phrase): g
            for (i, side, phrase), g in self.current.groundings.items()
            if i != operation_index
        }
        self.current = replace(
            self.current,
            plan=self.current.plan.model_copy(update={"operations": operations}),
            groundings=groundings,
        )
        self.acknowledged = shift_acknowledged(self.acknowledged, operation_index)
        return self._after_click("remove", str(operation_index + 1))

    # ------------------------------------------------------------ approving

    def approve(self) -> EditPlan:
        reasons = self.blockers()
        if reasons:
            raise ApprovalBlocked(reasons)
        assert self.current is not None

        self._name_tracks()
        self.approved = self.current.plan
        self._log("approve", one_line(self.current.confirmation))
        return self.approved

    def render(self, target: Path, *, preview: bool = True, **options):
        """Render the approved plan. Preview runs on the proxy -- same code path,
        a fraction of the time."""
        from cutpilot.editing import execute

        if self.approved is None:
            raise ApprovalBlocked(["승인되지 않은 계획은 렌더링하지 않습니다."])
        source = Path(self.asset.proxy_path) if preview else Path(self.asset.source_path)
        result = execute(
            self.approved, self.asset, self.tracks, self.boxes, target, source=source, **options
        )
        self._log("render", str(target))
        return result

    # -------------------------------------------------------------- internals

    def _operation(self, index: int) -> Operation:
        if self.current is None or not 0 <= index < len(self.current.plan.operations):
            raise IndexError(f"{index + 1}번 작업이 없습니다.")
        return self.current.plan.operations[index]

    def _check_known(self, track_ids: list[str]) -> None:
        unknown = [i for i in track_ids if i not in self.track_ids]
        if unknown:
            raise ValueError(f"영상에 없는 대상입니다: {', '.join(unknown)}")

    def _pending(self, index: int, side: str) -> list[GroundingKey]:
        assert self.current is not None
        return [
            key
            for key, g in self.current.groundings.items()
            if key[0] == index and key[1] == side and g.status == "choose"
        ]

    def _update_operation(self, index: int, **changes) -> None:
        assert self.current is not None
        operations = list(self.current.plan.operations)
        operations[index] = operations[index].model_copy(update=changes)
        # Re-validate: model_copy skips validators, and REFRAME has rules.
        operations[index] = Operation.model_validate(operations[index].model_dump())
        self.current = replace(
            self.current, plan=self.current.plan.model_copy(update={"operations": operations})
        )

    def _update_selector(self, index: int, **changes) -> None:
        op = self._operation(index)
        self._update_operation(index, selector=op.selector.model_copy(update=changes))

    def _replace(self, grounded: GroundedPlan) -> None:
        self.current = grounded
        self.approved = None

    def _after_click(self, kind: EventKind, detail: str, *, lead: str | None = None) -> Reply:
        assert self.current is not None
        confirmation = build_confirmation(
            self.current.plan,
            self.current.groundings,
            asset=self.asset,
            tracks=self.tracks,
            boxes=self.boxes,
            ui=self.ui,
        )
        self._replace(replace(self.current, confirmation=confirmation))
        # The planner has to know about clicks, or the next spoken correction
        # is made against a plan it has never seen.
        reply = self._reply(lead=lead)
        prefix = "[답변]" if kind == "answer" else "[화면 조작]"
        self.history.append(Turn("user", f"{prefix} {kind} {detail}"))
        self.history.append(Turn("agent", self._agent_turn(reply)))
        self._log(kind, detail)
        return reply

    def _reply(
        self,
        *,
        status: ReplyStatus = "plan",
        notes: list[str] | None = None,
        lead: str | None = None,
    ) -> Reply:
        confirmation = self.confirmation
        found = self.open_questions()
        question = found[0].question if found else None
        spec = self.spec
        if confirmation is None:
            text = lead or ""
        else:
            text = narrate(confirmation, question, len(found), spec, notes=notes, lead=lead)
        return Reply(
            status=status, text=text, confirmation=confirmation, notes=list(notes or []),
            question=question, remaining=len(found), spec=spec, blockers=self.blockers(),
        )  # fmt: skip

    @staticmethod
    def _agent_turn(reply: Reply) -> str:
        """What the planner sees of the agent's last turn: the plan in one line
        and the question it asked, so "the left one" can be read as an answer."""
        if reply.confirmation is None:
            return reply.text
        line = one_line(reply.confirmation)
        return f"{line} / 질문: {reply.question.text}" if reply.question else line

    def _plan_for_planner(self) -> EditPlan | None:
        """The current plan with open choices put back into words.

        A provisional pick is an id like any other; if the planner saw it, it
        would echo it back, the id would ground as settled, and the question
        would vanish without the user ever answering it.
        """
        if self.current is None:
            return None
        operations = []
        for index, op in enumerate(self.current.plan.operations):
            selector = op.selector
            for side in ("include", "exclude"):
                words: dict[str, str] = {}
                for key in self._pending(index, side):
                    grounding = self.current.groundings[key]
                    for track_id in grounding.ids_for_plan:
                        words[track_id] = grounding.phrase
                ids = [words.get(i, i) for i in getattr(selector, side)]
                ids = list(dict.fromkeys(i for i in ids if i))
                selector = selector.model_copy(update={side: ids})
            # model_construct: a reframe whose subject is still open has no
            # include, which the validator rightly rejects for a real plan.
            operations.append(Operation.model_construct(**{**dict(op), "selector": selector}))
        return EditPlan.model_construct(video_id=self.current.plan.video_id, operations=operations)

    def _remembering_grounder(self) -> Callable[..., Grounding]:
        """Ground a phrase once per session.

        A correction re-sends phrases that were already grounded ("the man in
        the cap ... only up to 15s"). Asking again costs a call and may pick
        someone else; an open choice should stay the same open choice.
        """
        previous: dict[tuple[str, str], Grounding] = {}
        if self.current is not None:
            for g in self.current.groundings.values():
                if g.phrase and track_id_of(g.phrase) is None:
                    previous[(g.phrase, g.kind)] = g

        def ground(provider, phrase, *, kind, **kwargs) -> Grounding:
            target = "license_plate" if kind == "license_plate" else "person"
            if (hit := previous.get((phrase, target))) is not None:
                return replace(hit, chosen=list(hit.chosen), candidates=list(hit.candidates))
            return ground_phrase(provider, phrase, kind=kind, **kwargs)

        return ground

    def _name_tracks(self) -> None:
        """Remember what the user called someone, so the next instruction can
        refer to them by it ("the man in the cap" stays person_03)."""
        assert self.current is not None
        by_id = {t.id: t for t in self.tracks}
        for g in self.current.groundings.values():
            if len(g.chosen) == 1 and g.phrase and track_id_of(g.phrase) is None:
                track = by_id.get(g.chosen[0])
                if track is not None:
                    track.display_name = g.phrase

    def _log(self, kind: EventKind, detail: str, calls_before: int | None = None) -> None:
        calls = 0 if calls_before is None else self.provider.calls - calls_before
        self.events.append(SessionEvent(kind=kind, detail=detail, llm_calls=calls))


def _error_text(exc: LLMError) -> str:
    if isinstance(exc, LLMBudgetExceeded):
        return (
            f"설정한 AI 사용 예산(${exc.budget:.2f})을 모두 썼습니다. "
            "화면에서 직접 고치는 건 계속 할 수 있습니다."
        )
    if isinstance(exc, LLMQuotaExhausted):
        return (
            "오늘 사용할 수 있는 AI 호출을 모두 썼습니다 "
            f"(약 {exc.resets_in_hours:.0f}시간 뒤 초기화). "
            "화면에서 직접 고치는 건 계속 할 수 있습니다."
        )
    return "AI 응답을 받지 못했습니다. 잠시 뒤 다시 말씀해 주세요. 지금 계획은 그대로입니다."
