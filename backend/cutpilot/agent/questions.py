"""Turn a confirmation into the questions the agent stops to ask, and the spec.

The confirmation already says where the agent is unsure (per-field status) and
what a render would hide (warnings). This module only decides how to raise
those with the user: one at a time, in plan order, most blocking first, and
never twice once answered.

Two kinds of question exist and they are treated differently:

    blocking      choose, missing. The agent will not decide these for the
                  user, so no option can wave them through -- only a pick, a
                  rephrase, or dropping the operation settles them.
    non-blocking  confirm, track_gap, out_of_frame. The agent has a sensible
                  default but the user should see it before a render does.
                  These have one-click answers and stay answered.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cutpilot.agent.confirm import KIND_LABEL, OP_LABEL, GroundedPlan
from cutpilot.agent.grounder import is_unresolved
from cutpilot.agent.planner import track_id_of
from cutpilot.schemas.plan import EditPlan
from cutpilot.schemas.question import (
    QUESTION_ORDER,
    Question,
    QuestionOption,
    Spec,
    SpecStep,
)
from cutpilot.schemas.track import Track
from cutpilot.vision.resolve import resolve_selector

YES = QuestionOption(id="yes", label="맞아요")
NO = QuestionOption(id="no", label="아니에요, 다시 고를게요")
FILL = QuestionOption(id="fill", label="그 구간도 처리")
LEAVE = QuestionOption(id="leave", label="그대로 두기")
KEEP = QuestionOption(id="keep", label="그대로 유지")
REMOVE = QuestionOption(id="remove", label="이 작업 빼기")
OK = QuestionOption(id="ok", label="확인")


@dataclass
class OpenQuestion:
    question: Question
    key: tuple
    """What the answer acts on: a grounding key for choose/confirm, the track
    and gap for track_gap, the operation index otherwise."""

    ack: tuple
    """(kind, operation index, ...what it is about). Remembered once answered,
    so a later replan that leaves the same thing in place does not ask again.
    The index is kept separately so dropping an operation can shift it."""


def _qid(ack: tuple) -> str:
    return hashlib.sha1("|".join(map(str, ack)).encode()).hexdigest()[:10]


def _ask(ack: tuple, key: tuple, **fields) -> OpenQuestion:
    return OpenQuestion(
        Question(id=_qid(ack), kind=ack[0], operation_index=ack[1], **fields), key=key, ack=ack
    )


def _step_name(plan: EditPlan, index: int) -> str:
    op = plan.operations[index]
    return f"{index + 1}번 작업({OP_LABEL[op.op]})"


def open_questions(current: GroundedPlan | None, acknowledged: set[tuple]) -> list[OpenQuestion]:
    """Everything still worth asking, in the order it should be asked."""
    if current is None:
        return []
    plan, confirmation, groundings = current.plan, current.confirmation, current.groundings
    found: list[OpenQuestion] = []

    for index in range(len(plan.operations)):
        name = _step_name(plan, index)
        own = {k: g for k, g in groundings.items() if k[0] == index}

        choosing = [k for k, g in own.items() if g.status == "choose"]
        if choosing:
            item = next(i for i in confirmation.items_for(index) if i.field == "selector")
            phrases = [k[2] for k in choosing if k[2]]
            if not phrases:
                text = f"{name}: 화면이 따라갈 사람을 정해 주세요. 후보 중에서 골라 주세요."
            elif item.candidates:
                text = (
                    f"{name}: '{', '.join(phrases)}'이(가) 누구인지 확실하지 않습니다. "
                    "화면의 후보 중에서 골라 주세요."
                )
            else:
                text = (
                    f"{name}: '{', '.join(phrases)}'을(를) 화면에서 찾지 못했습니다. "
                    "다르게 설명하거나 화면에서 직접 골라 주세요."
                )
            t = next((own[k].t for k in choosing if own[k].t is not None), None)
            found.append(
                _ask(
                    ("choose", index, *phrases),
                    choosing[0],
                    text=text,
                    blocking=True,
                    options=[REMOVE],
                    candidates=item.candidates,
                    seek_t=t,
                )  # fmt: skip
            )
            # Gaps and crops of a provisional target are not worth asking about yet.
            continue

        empty = [
            w for w in confirmation.warnings
            if w.kind == "empty_selection" and w.operation_index == index
        ]  # fmt: skip
        if empty:
            found.append(
                _ask(
                    ("missing", index, empty[0].message),
                    (index,),
                    blocking=True,
                    options=[REMOVE],
                    text=f"{name}: {empty[0].message} 다르게 설명해 주시거나 이 작업을 뺄 수 "
                    "있습니다.",
                )  # fmt: skip
            )
            continue

        for key, g in own.items():
            if g.status != "confirm":
                continue
            kind = KIND_LABEL.get(g.kind, g.kind)
            side = "제외할 대상" if key[1] == "exclude" else "대상"
            found.append(
                _ask(
                    ("confirm", index, key[1], g.phrase, *g.chosen),
                    key,
                    blocking=False,
                    options=[YES, NO],
                    candidates=list(g.chosen),
                    seek_t=g.t,
                    text=f"{name}: '{g.phrase}'을(를) {', '.join(g.chosen)}({kind})(으)로 "
                    f"찾았습니다. 이 {side}이 맞나요?",
                )  # fmt: skip
            )

        for w in confirmation.warnings:
            if w.operation_index != index or w.kind not in ("track_gap", "out_of_frame"):
                continue
            if w.kind == "track_gap":
                assert w.time_range is not None and w.track_id is not None
                gap = (w.track_id, w.time_range.start, w.time_range.end)
                found.append(
                    _ask(
                        ("track_gap", index, *gap),
                        (index, *gap),
                        blocking=False,
                        options=[FILL, LEAVE],
                        track_id=w.track_id,
                        time_range=w.time_range,
                        seek_t=w.time_range.start,
                        text=f"{name}: {w.message} 가려진 거라면 앞뒤 위치를 이어서 "
                        "그 구간도 처리할까요?",
                    )  # fmt: skip
                )
            else:
                found.append(
                    _ask(
                        ("out_of_frame", index, w.message),
                        (index,),
                        blocking=False,
                        options=[KEEP, REMOVE],
                        text=f"{name}: {w.message} 그래도 유지할까요?",
                    )  # fmt: skip
                )

    for w in confirmation.warnings:
        if w.operation_index is None:  # an operation that could not be built at all
            found.append(
                _ask(
                    ("missing", None, w.message),
                    (),
                    blocking=False,
                    options=[OK],
                    text=f"{w.message} 이 작업은 빼고 진행합니다.",
                )  # fmt: skip
            )

    found = [q for q in found if q.question.blocking or q.ack not in acknowledged]
    found.sort(
        key=lambda q: (
            -1 if q.question.operation_index is None else q.question.operation_index,
            QUESTION_ORDER[q.question.kind],
        )
    )
    return found


def shift_acknowledged(acknowledged: set[tuple], removed: int) -> set[tuple]:
    """Answers about later operations move up when one is dropped."""
    out: set[tuple] = set()
    for ack in acknowledged:
        index = ack[1]
        if index is None or index < removed:
            out.add(ack)
        elif index > removed:
            out.add((ack[0], index - 1, *ack[2:]))
    return out


# --------------------------------------------------------------------- spec


def _named(track_id: str, names: dict[str, str]) -> str:
    return f"{track_id} ({names[track_id]})" if track_id in names else track_id


def build_spec(
    current: GroundedPlan, *, tracks: list[Track], duration: float, decisions: list[str]
) -> Spec:
    """The plan written out the way it will run: who, what, when."""
    names = {t.id: t.display_name for t in tracks if t.display_name}
    for g in current.groundings.values():
        if len(g.chosen) == 1 and g.phrase and track_id_of(g.phrase) is None:
            names.setdefault(g.chosen[0], g.phrase)

    steps: list[SpecStep] = []
    lines = ["최종 명세서"]
    for index, op in enumerate(current.plan.operations):
        label = OP_LABEL[op.op] + (f" {op.aspect_ratio}" if op.aspect_ratio else "")
        kind = KIND_LABEL.get(op.selector.kind, op.selector.kind)
        targets = [t.id for t in resolve_selector(op.selector, tracks)]
        excluded = [i for i in op.selector.exclude if not is_unresolved(i)]
        time = op.time_range.describe(duration)
        steps.append(
            SpecStep(
                index=index,
                op=op.op,
                label=label,
                time=time,
                targets=[_named(i, names) for i in targets],
                excluded=[_named(i, names) for i in excluded],
            )  # fmt: skip
        )
        if op.op == "REFRAME":
            who = f"기준 인물 {_named(targets[0], names)}" if targets else "기준 인물 없음"
        elif op.selector.include:
            who = f"{kind}: " + ", ".join(_named(i, names) for i in targets)
        else:
            who = f"{kind} 전체 ({len(targets)}개)"
            if excluded:
                who += ", 제외: " + ", ".join(_named(i, names) for i in excluded)
        lines.append(f"{index + 1}. {label} — {who} — {time}")

    if decisions:
        lines.append("확인한 내용:")
        lines += [f"- {d}" for d in decisions]
    lines.append("이대로 실행할까요? 승인하면 렌더링합니다.")
    return Spec(steps=steps, decisions=list(decisions), text="\n".join(lines))
