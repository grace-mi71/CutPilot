# ruff: noqa: F811  -- fixtures imported from test_session are pytest parameters
"""Plan once, answer the agent's questions one at a time, approve the spec.

Same scripted model and scene as test_session: "빨간 옷" is one clear match
(confirm), "모자 쓴 사람" two (choose), person_03 is out of sight 4-6s (gap).
"""

from __future__ import annotations

import pytest

from cutpilot.agent.session import ApprovalBlocked, quick_option
from tests.test_grounding import asset, boxes, tracks  # noqa: F401
from tests.test_session import Model, mosaic, session_for, settle  # noqa: F401


def plan(*steps: dict) -> dict:
    return {"operations": [op for step in steps for op in step["operations"]]}


def blur(*phrases: str, **kw) -> dict:
    step = mosaic(*phrases, **kw)
    step["operations"][0]["op"] = "BLUR"
    return step


class TestOrder:
    def test_questions_follow_the_plan_one_at_a_time(self, session_for):
        model = Model(plan(blur("빨간 옷"), mosaic("모자 쓴 사람")))
        session = session_for(model)
        reply = session.say("1. 빨간 옷 블러 2. 모자 쓴 사람 모자이크")

        assert len(session.plan.operations) == 2
        assert (reply.question.kind, reply.question.operation_index) == ("confirm", 0)
        assert reply.remaining == 2

        reply = session.say("네")
        assert (reply.question.kind, reply.question.operation_index) == ("choose", 1)
        assert reply.question.blocking
        assert set(reply.question.candidates) == {"person_01", "person_03"}

        reply = session.choose(1, ["person_01"])
        assert reply.question is None
        assert [s.op for s in reply.spec.steps] == ["BLUR", "MOSAIC"]
        assert reply.spec.decisions == [
            "'빨간 옷' = person_02 (확인)",
            "'모자 쓴 사람' → person_01",
        ]
        assert reply.can_approve
        assert len(model.requests) == 3  # one plan, two groundings -- answers were free

    def test_a_blocking_question_holds_back_checks_on_its_guess(self, session_for):
        session = session_for(Model(mosaic("모자 쓴 사람")))
        session.say("모자 쓴 사람 모자이크")
        assert [q.question.kind for q in session.open_questions()] == ["choose"]

    def test_unanswered_check_blocks_approval(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        with pytest.raises(ApprovalBlocked, match="질문"):
            session.approve()


class TestAnswers:
    def test_no_turns_a_guess_into_a_choice(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        reply = session.say("아니요")

        assert reply.question.kind == "choose"
        assert "person_02" not in reply.question.candidates
        assert reply.question.candidates  # the others on screen
        assert not reply.can_approve
        session.choose(0, ["person_01"])
        assert session.plan.operations[0].selector.include == ["person_01"]

    def test_bare_yes_is_free_but_a_correction_is_not(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        q = session.question
        assert quick_option("네!", q) == "yes"
        assert quick_option("아니", q) == "no"
        assert quick_option("아니 5초부터", q) is None

    def test_filling_a_gap_covers_it(self, session_for):
        session = session_for(Model(mosaic("person_03")))
        reply = session.say("person_03 모자이크")
        assert reply.question.kind == "track_gap"

        reply = session.answer(reply.question.id, "fill")
        track = next(t for t in session.tracks if t.id == "person_03")
        assert track.present_at(5.0)
        assert session.boxes["person_03"].at(5.0) is not None
        assert not [w for w in reply.confirmation.warnings if w.kind == "track_gap"]
        assert "가려진 구간도" in reply.spec.decisions[0]

    def test_gap_fill_does_not_touch_the_shared_analysis(self, session_for, tracks, boxes):
        session = session_for(Model(mosaic("person_03")))
        session.say("person_03 모자이크")
        session.answer(session.question.id, "fill")
        assert boxes["person_03"].at(5.0) is None
        assert not next(t for t in tracks if t.id == "person_03").present_at(5.0)

    def test_an_answer_is_not_asked_again_after_a_correction(self, session_for):
        model = Model(mosaic("person_03"), blur("person_03"))
        session = session_for(model)
        session.say("person_03 모자이크")
        session.answer(session.question.id, "leave")
        reply = session.say("블러로 바꿔")
        assert reply.question is None
        assert reply.can_approve

    def test_removing_shifts_later_answers(self, session_for):
        model = Model(plan(mosaic("유령"), mosaic("person_03")))
        session = session_for(model)
        session.say("유령이랑 person_03 모자이크")
        gap = next(q for q in session.open_questions() if q.question.kind == "track_gap")
        session.answer(gap.question.id, "leave")
        blocking = session.question
        assert blocking.operation_index == 0
        reply = session.answer(blocking.id, "remove")
        assert len(session.plan.operations) == 1
        assert reply.question is None  # the gap answer moved with its operation

    def test_stale_question_is_rejected(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        q = session.question
        session.answer(q.id, "yes")
        with pytest.raises(IndexError):
            session.answer(q.id, "yes")
        with pytest.raises(IndexError):
            session.answer("nope", "yes")

    def test_unknown_option_is_rejected(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        with pytest.raises(ValueError):
            session.answer(session.question.id, "fill")

    def test_the_planner_sees_the_question_it_is_answering(self, session_for):
        model = Model(mosaic("모자 쓴 사람"), mosaic("빨간 옷"))
        session = session_for(model)
        session.say("모자 쓴 사람 모자이크")
        session.say("빨간 옷 입은 사람이요")
        assert "질문: 1번 작업(모자이크): '모자 쓴 사람'" in model.planner_prompts[1]
