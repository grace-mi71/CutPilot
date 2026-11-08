"""Score a plan against what the user meant.

The privacy measure is in seconds of screen time, not in ids:

    missed_s   how long someone who should be covered is on screen uncovered --
               an exposure, the failure that makes a privacy edit unusable
    extra_s    how long someone who should not be touched is covered anyway

Both are measured over track presence at a fixed step, so a wrong time range
and a wrong person are penalised in the same unit, and an id-level mistake that
happens not to matter on screen (say, a target that has already left) costs
nothing. Reframes are scored on the subject and the ratio.

A task succeeds when every intended operation is there, in order, of the right
kind, and costs no more than `TOLERANCE_S` either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cutpilot.eval.tasks import ExpectedOp, Task
from cutpilot.schemas.plan import EditPlan, Operation
from cutpilot.schemas.track import Track
from cutpilot.vision.resolve import resolve_selector, translate_to_kind

STEP_S = 0.04
"""One frame at 25 fps."""

TOLERANCE_S = 0.25
"""Rounding of a spoken "3 seconds" against a track boundary is not an error."""


@dataclass
class OpScore:
    expected: str
    got: str | None
    op_ok: bool
    targets_ok: bool
    missed_s: float = 0.0
    extra_s: float = 0.0
    note: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.op_ok
            and self.targets_ok
            and self.missed_s <= TOLERANCE_S
            and self.extra_s <= TOLERANCE_S
        )


@dataclass
class TaskScore:
    ops: list[OpScore] = field(default_factory=list)
    extra_ops: int = 0

    @property
    def success(self) -> bool:
        return bool(self.ops) and all(o.ok for o in self.ops) and self.extra_ops == 0

    @property
    def missed_s(self) -> float:
        return round(sum(o.missed_s for o in self.ops), 2)

    @property
    def extra_s(self) -> float:
        return round(sum(o.extra_s for o in self.ops), 2)


def expected_tracks(exp: ExpectedOp, tracks: list[Track]) -> set[str]:
    pool = [t.id for t in tracks if t.kind == exp.kind]
    wanted = set(pool) if exp.targets == "all" else set(exp.targets) & set(pool)
    return wanted - set(exp.exclude)


def selected_as(op: Operation, kind: str, tracks: list[Track]) -> set[str]:
    """Who an operation touches, named in the expected kind.

    Face or body is not scored: "blur the man in the cap" is met by blurring
    his face or his whole box. Who is covered, and when, is what is scored.
    """
    out: set[str] = set()
    for track in resolve_selector(op.selector, tracks):
        out.update([track.id] if track.kind == kind else translate_to_kind(track.id, kind, tracks))
    return out


def _coverage(ids: set[str], lo: float, hi: float, tracks: list[Track]) -> set[tuple[str, int]]:
    by_id = {t.id: t for t in tracks}
    out: set[tuple[str, int]] = set()
    for track_id in ids:
        track = by_id.get(track_id)
        if track is None:
            continue
        n = int(lo / STEP_S)
        while n * STEP_S < hi:
            if n * STEP_S >= lo and track.present_at(n * STEP_S):
                out.add((track_id, n))
            n += 1
    return out


def score_op(
    exp: ExpectedOp, got: Operation | None, tracks: list[Track], duration: float
) -> OpScore:
    label = f"{exp.op} {exp.targets if exp.targets == 'all' else ','.join(exp.targets)}"
    if got is None:
        want = expected_tracks(exp, tracks)
        lo, hi = exp.time_range.resolve(duration)
        missed = len(_coverage(want, lo, hi, tracks)) * STEP_S if exp.op != "REFRAME" else 0.0
        return OpScore(label, None, False, False, missed_s=round(missed, 2), note="missing")

    got_ids = selected_as(got, exp.kind, tracks)
    summary = f"{got.op} {','.join(sorted(got_ids)) or '-'}"
    op_ok = got.op == exp.op

    if exp.op == "REFRAME":
        want = set(exp.targets) if exp.targets != "all" else set()
        ratio_ok = (got.aspect_ratio or "9:16") == (exp.aspect_ratio or "9:16")
        return OpScore(label, summary, op_ok and ratio_ok, got_ids == want)

    want = expected_tracks(exp, tracks)
    e_lo, e_hi = exp.time_range.resolve(duration)
    g_lo, g_hi = got.time_range.resolve(duration)
    should = _coverage(want, e_lo, e_hi, tracks)
    does = _coverage(got_ids, g_lo, g_hi, tracks) if op_ok else set()
    return OpScore(
        label, summary, op_ok,
        targets_ok=got_ids == want,
        missed_s=round(len(should - does) * STEP_S, 2),
        extra_s=round(len(does - should) * STEP_S, 2),
    )  # fmt: skip


def score_plan(
    task: Task, plan: EditPlan | None, tracks: list[Track], duration: float
) -> TaskScore:
    """Operations are matched in order: a step the user listed second should be
    the agent's second step too."""
    got = list(plan.operations) if plan else []
    score = TaskScore()
    for i, exp in enumerate(task.expected):
        score.ops.append(score_op(exp, got[i] if i < len(got) else None, tracks, duration))
    score.extra_ops = max(0, len(got) - len(task.expected))
    return score
