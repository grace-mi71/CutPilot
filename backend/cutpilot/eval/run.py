"""Run tasks under two conditions and score both.

    auto        the agent's first plan, executed as is -- no questions, no review.
                Provisional guesses stand; an open choice stays whatever the
                agent guessed. This is "no confirmation".
    negotiated  the same first plan, then a simulated user who knows the answer
                works through the agent's questions and reviews the spec,
                fixing what is visibly wrong by clicking -- never by typing, so
                the negotiation costs no extra model calls.

Both conditions share one model turn, so the difference between them is the
negotiation alone, measured in accuracy gained against questions answered and
clicks made. The simulated user is an upper bound on what a careful person
would achieve; how close real people come, and whether seeing the boxes helps
them (text-only vs text+visual), is what the user study is for.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from cutpilot.agent.session import EditSession
from cutpilot.eval.score import (
    TOLERANCE_S,
    TaskScore,
    expected_tracks,
    score_op,
    score_plan,
    selected_as,
)
from cutpilot.eval.tasks import ExpectedOp, Task
from cutpilot.schemas.track import Track

MAX_STEPS = 30


@dataclass
class RunRecord:
    task: str
    level: str
    condition: str
    status: str
    success: bool
    missed_s: float
    extra_s: float
    questions: int = 0
    """Questions the agent asked and the user answered."""
    corrections: int = 0
    """Clicks the user made on their own, reviewing the spec."""
    unfixable: list[str] = field(default_factory=list)
    """What the user could see was wrong but could not fix by clicking."""
    llm_calls: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    ops: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _record(task: Task, condition: str, status: str, score: TaskScore, **kw) -> RunRecord:
    return RunRecord(
        task=task.id, level=task.level, condition=condition, status=status,
        success=score.success, missed_s=score.missed_s, extra_s=score.extra_s,
        ops=[asdict(o) | {"ok": o.ok} for o in score.ops], **kw,
    )  # fmt: skip


# ------------------------------------------------------------------ oracle


class Oracle:
    """A user who knows what they meant and answers truthfully."""

    def __init__(self, session: EditSession, task: Task) -> None:
        self.session = session
        self.task = task
        self.questions = 0
        self.corrections = 0
        self.unfixable: list[str] = []
        self.log: list[str] = []

    def _expected(self, index: int | None) -> ExpectedOp | None:
        if index is None or index >= len(self.task.expected):
            return None
        return self.task.expected[index]

    def _truth(self, exp: ExpectedOp, side: str) -> list[str]:
        if side == "exclude":
            return list(exp.exclude)
        if exp.targets == "all":
            return []
        return list(exp.targets)

    def answer(self) -> bool:
        """Answer the agent's current question. False when there is none."""
        found = self.session.open_questions()
        if not found:
            return False
        oq = found[0]
        q = oq.question
        exp = self._expected(q.operation_index)
        index = q.operation_index
        op = self.session.plan.operations[index] if index is not None else None
        self.questions += 1

        if q.kind in ("choose", "missing") and (exp is None or op is None or op.op != exp.op):
            if q.operation_index is None:
                self.session.answer(q.id, "ok")
                self.log.append("ok (dropped step)")
            else:
                self.session.answer(q.id, "remove")
                self.log.append(f"remove step {q.operation_index + 1}")
            return True

        if q.kind == "choose":
            side = oq.key[1]
            truth = self._truth(exp, side)
            if exp.op == "REFRAME":
                truth = truth[:1]
            if not truth:
                self.session.answer(q.id, "remove")
                self.log.append(f"remove step {q.operation_index + 1} (wanted everyone)")
                return True
            self.session.choose(q.operation_index, truth, side=side)
            self.log.append(f"choose {truth}")
        elif q.kind == "missing":
            self.session.answer(q.id, "remove")
            self.log.append(f"remove step {q.operation_index + 1} (nothing found)")
        elif q.kind == "confirm":
            grounding = self.session.current.groundings[oq.key]
            truth = set(self._truth(exp, oq.key[1])) if exp else set()
            option = "yes" if grounding.chosen and set(grounding.chosen) <= truth else "no"
            self.session.answer(q.id, option)
            self.log.append(f"confirm {grounding.chosen}: {option}")
        else:  # gap: the user cannot know better than the agent; crop: keep
            option = "leave" if q.kind == "track_gap" else q.options[0].id
            self.session.answer(q.id, option)
            self.log.append(f"{q.kind}: {option}")
        return True

    def review(self, tracks: list[Track], duration: float) -> bool:
        """Read the spec and fix one visible mistake by clicking. False when
        nothing more can be fixed that way."""
        plan = self.session.plan
        if plan is None:
            return False
        for i, op in enumerate(plan.operations):
            exp = self._expected(i)
            if exp is None:
                self.session.remove(i)
                self._fixed(f"remove extra step {i + 1}")
                return True
            if op.op != exp.op:
                self._cannot(f"step {i + 1} is {op.op}, wanted {exp.op}")
                continue

            got = selected_as(op, exp.kind, tracks)
            want = expected_tracks(exp, tracks) if exp.op != "REFRAME" else set(exp.targets[:1])
            if got == want:
                # Right people; the time only matters where it changes what is
                # covered on screen, which is what a user watching would see.
                s = score_op(exp, op, tracks, duration)
                if s.missed_s > TOLERANCE_S or s.extra_s > TOLERANCE_S:
                    self.session.set_time_range(i, exp.start, exp.end)
                    self._fixed(f"step {i + 1} time -> {exp.start}~{exp.end}")
                    return True
                continue
            if exp.op == "REFRAME":
                self.session.toggle(i, next(iter(want)))
                self._fixed(f"step {i + 1} subject -> {want}")
                return True
            whole_kind = not op.selector.include
            if whole_kind != (exp.targets == "all"):
                self._cannot(f"step {i + 1} targets {sorted(got)}, wanted {sorted(want)}")
                continue
            # Add what is missing before dropping what is extra: the last
            # explicit target cannot be dropped.
            for track_id in sorted(want - got) + sorted(got - want):
                self.session.toggle(i, track_id)
                self._fixed(f"step {i + 1} toggle {track_id}")
                return True
        for j in range(len(plan.operations), len(self.task.expected)):
            self._cannot(f"step {j + 1} ({self.task.expected[j].op}) is missing")
        return False

    def _fixed(self, what: str) -> None:
        self.corrections += 1
        self.log.append(what)

    def _cannot(self, what: str) -> None:
        if what not in self.unfixable:
            self.unfixable.append(what)


# ------------------------------------------------------------------ runner


def run_task(
    task: Task,
    make_session: Callable[[], EditSession],
    *,
    tracks: list[Track],
    duration: float,
    spent: Callable[[], float] = lambda: 0.0,
) -> list[RunRecord]:
    session = make_session()
    usd0, t0 = spent(), time.perf_counter()
    reply = session.say(task.text, ui=task.ui)
    calls, usd, seconds = session.provider.calls, spent() - usd0, time.perf_counter() - t0
    shared = {"llm_calls": calls, "usd": round(usd, 5), "seconds": round(seconds, 2)}

    first = session.plan if reply.status == "plan" else None
    auto = _record(
        task, "auto", reply.status, score_plan(task, first, tracks, duration),
        log=[reply.text], **shared,
    )  # fmt: skip
    if reply.status != "plan":
        negotiated = _record(
            task, "negotiated", reply.status, score_plan(task, None, tracks, duration),
            log=[reply.text], **shared,
        )  # fmt: skip
        return [auto, negotiated]

    oracle = Oracle(session, task)
    for _ in range(MAX_STEPS):
        if not oracle.answer() and not oracle.review(tracks, duration):
            break
    if session.provider.calls != calls:  # the oracle only clicks; this would be a bug
        raise AssertionError("negotiation made a model call")

    negotiated = _record(
        task, "negotiated", "plan", score_plan(task, session.plan, tracks, duration),
        questions=oracle.questions, corrections=oracle.corrections,
        unfixable=oracle.unfixable, log=oracle.log, **shared,
    )  # fmt: skip
    return [auto, negotiated]


def summarize(records: list[RunRecord]) -> list[dict]:
    """One row per (level, condition), plus an "all" row per condition.

    `needless_q` counts questions asked on tasks the agent already had right
    without asking -- the price of the negotiation where it bought nothing.
    """
    right_first = {r.task for r in records if r.condition == "auto" and r.success}
    rows = []
    levels = sorted({r.level for r in records}) + ["all"]
    for level in levels:
        for condition in ("auto", "negotiated"):
            group = [
                r for r in records
                if r.condition == condition and (level == "all" or r.level == level)
            ]  # fmt: skip
            if not group:
                continue
            n = len(group)
            rows.append(
                {
                    "level": level,
                    "condition": condition,
                    "tasks": n,
                    "success": round(sum(r.success for r in group) / n, 2),
                    "missed_s": round(sum(r.missed_s for r in group), 2),
                    "extra_s": round(sum(r.extra_s for r in group), 2),
                    "questions": round(sum(r.questions for r in group) / n, 2),
                    "corrections": round(sum(r.corrections for r in group) / n, 2),
                    "needless_q": sum(r.questions for r in group if r.task in right_first),
                    "llm_calls": round(sum(r.llm_calls for r in group) / n, 2),
                }
            )
    return rows
