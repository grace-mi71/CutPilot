# ruff: noqa: F811  -- fixtures imported from test_session are pytest parameters
"""Scoring and the simulated user, on the three-person scene.

person_01 left, person_02 middle, person_03 right and absent 4-6s; 10s long.
"""

from __future__ import annotations

from pathlib import Path

from cutpilot.eval.run import run_task, summarize
from cutpilot.eval.score import score_plan
from cutpilot.eval.tasks import Task, TaskSet
from cutpilot.schemas.plan import EditPlan, Operation, Selector, TimeRange
from tests.test_grounding import DURATION, asset, boxes, tracks  # noqa: F401
from tests.test_session import Model, mosaic, session_for  # noqa: F401

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def task(expected: list[dict], text: str = "x", level: str = "L2") -> Task:
    return Task(id="t", level=level, text=text, expected=expected)


def plan(*ops: Operation) -> EditPlan:
    return EditPlan(video_id="v", operations=list(ops))


def mosaic_op(*include: str, start=None, end=None, exclude=()) -> Operation:
    return Operation(
        op="MOSAIC", selector=Selector(kind="person", include=list(include), exclude=list(exclude)),
        time_range=TimeRange(start=start, end=end),
    )  # fmt: skip


class TestScore:
    def test_exact_plan_succeeds(self, tracks):
        t = task([{"op": "MOSAIC", "targets": ["person_02"]}])
        assert score_plan(t, plan(mosaic_op("person_02")), tracks, DURATION).success

    def test_wrong_person_is_exposure_and_overreach(self, tracks):
        t = task([{"op": "MOSAIC", "targets": ["person_02"]}])
        s = score_plan(t, plan(mosaic_op("person_01")), tracks, DURATION)
        assert not s.success
        assert s.missed_s == DURATION and s.extra_s == DURATION

    def test_absence_is_not_exposure(self, tracks):
        """person_03 is off screen 4-6s: stopping the blur there costs nothing."""
        t = task([{"op": "MOSAIC", "targets": ["person_03"], "end": 6.0}])
        s = score_plan(t, plan(mosaic_op("person_03", end=4.0)), tracks, DURATION)
        assert s.success and s.missed_s == 0.0

    def test_short_range_is_measured_in_seconds(self, tracks):
        t = task([{"op": "MOSAIC", "targets": ["person_02"]}])
        s = score_plan(t, plan(mosaic_op("person_02", end=7.0)), tracks, DURATION)
        assert s.missed_s == 3.0 and not s.success

    def test_all_but_one(self, tracks):
        t = task([{"op": "MOSAIC", "targets": "all", "exclude": ["person_01"]}])
        s = score_plan(t, plan(mosaic_op(exclude=["person_01"])), tracks, DURATION)
        assert s.success

    def test_missing_and_extra_steps(self, tracks):
        t = task([{"op": "MOSAIC", "targets": ["person_02"]}])
        assert score_plan(t, None, tracks, DURATION).missed_s == DURATION
        two = plan(mosaic_op("person_02"), mosaic_op("person_01"))
        extra = score_plan(t, two, tracks, DURATION)
        assert extra.extra_ops == 1 and not extra.success


class TestRun:
    def run(self, session_for, tracks, model, t):
        return run_task(t, lambda: session_for(model), tracks=tracks, duration=DURATION)

    def test_negotiation_fixes_an_ambiguous_guess(self, session_for, tracks):
        """'모자 쓴 사람' matches person_01 and person_03; the user meant person_03."""
        t = task([{"op": "MOSAIC", "targets": ["person_03"]}], level="L3")
        auto, neg = self.run(session_for, tracks, Model(mosaic("모자 쓴 사람")), t)
        assert not auto.success
        assert neg.success
        assert neg.questions >= 1  # the choice, then the gap
        assert neg.llm_calls == auto.llm_calls == 2  # negotiating cost no model calls

    def test_confirmed_guess_needs_one_yes(self, session_for, tracks):
        t = task([{"op": "MOSAIC", "targets": ["person_02"]}])
        auto, neg = self.run(session_for, tracks, Model(mosaic("빨간 옷")), t)
        assert auto.success and neg.success
        assert (neg.questions, neg.corrections) == (1, 0)

    def test_review_fixes_the_time_by_clicking(self, session_for, tracks):
        t = task([{"op": "MOSAIC", "targets": ["person_02"], "end": 5.0}])
        auto, neg = self.run(session_for, tracks, Model(mosaic("빨간 옷")), t)
        assert not auto.success and neg.success
        assert neg.corrections == 1

    def test_wrong_op_cannot_be_clicked_away(self, session_for, tracks):
        t = task([{"op": "BLUR", "targets": ["person_02"]}])
        _, neg = self.run(session_for, tracks, Model(mosaic("빨간 옷")), t)
        assert not neg.success
        assert neg.unfixable

    def test_summary_rows(self, session_for, tracks):
        t = task([{"op": "MOSAIC", "targets": ["person_02"]}])
        rows = summarize(self.run(session_for, tracks, Model(mosaic("빨간 옷")), t))
        assert {(r["level"], r["condition"]) for r in rows} == {
            ("L2", "auto"), ("L2", "negotiated"), ("all", "auto"), ("all", "negotiated"),
        }  # fmt: skip


def test_crowd_task_set_is_valid():
    s = TaskSet.load(EVAL_DIR / "crowd" / "tasks.json")
    assert {t.level for t in s.tasks} == {"L1", "L2", "L3", "L4"}
    assert len({t.id for t in s.tasks}) == len(s.tasks)


def test_face_edit_of_the_right_person_counts(tracks):
    """face_01 belongs to person_01 in the scene: blurring the face meets the task."""
    t = task([{"op": "MOSAIC", "targets": ["person_01"]}])
    op = Operation(op="MOSAIC", selector=Selector(kind="face", include=["person_01"]))
    assert score_plan(t, plan(op), tracks, DURATION).success
