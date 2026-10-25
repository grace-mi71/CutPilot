"""The negotiation loop, end to end with a scripted model.

Same three-person scene as the grounding tests. The fake model answers the
planner from a script and grounds phrases from a fixed mapping:

    "빨간 옷"      -> person_02             one match: confirm
    "모자 쓴 사람"  -> person_01, person_03  two matches: choose
    "유령"         -> nobody                 unresolved
"""

from __future__ import annotations

import re

import pytest

from cutpilot.agent.llm import FakeProvider, LLMQuotaExhausted, LLMUnavailable
from cutpilot.agent.prompts import PLANNER_SYSTEM
from cutpilot.agent.session import ApprovalBlocked, EditSession
from cutpilot.schemas.session import UIContext
from tests.test_grounding import FRAME, _gemini_box, asset, boxes, tracks  # noqa: F401

SEES = {
    "빨간 옷": ["person_02"],
    "모자 쓴 사람": ["person_01", "person_03"],
    "유령": [],
}


def mosaic(*phrases: str, kind: str = "person", exclude=(), start=None, end=None) -> dict:
    return {
        "operations": [
            {
                "op": "MOSAIC",
                "selector": {
                    "kind": kind,
                    "include_phrases": list(phrases),
                    "exclude_phrases": list(exclude),
                },
                "time_range": {"start": start, "end": end},
            }
        ]
    }


class Model(FakeProvider):
    """Planner answers come from `plans` in order; grounding from SEES."""

    def __init__(self, *plans) -> None:
        super().__init__(responder=self._respond)
        self.plans = list(plans)
        self.failure: Exception | None = None

    def _respond(self, request):
        if self.failure is not None:
            raise self.failure
        if request.system == PLANNER_SYSTEM:
            return self.plans.pop(0)
        phrase = re.search(r'"(.+?)"', request.prompt).group(1)
        return {"objects": [{"box_2d": _gemini_box(i), "label": phrase} for i in SEES[phrase]]}

    @property
    def planner_prompts(self) -> list[str]:
        return [r.prompt for r in self.requests if r.system == PLANNER_SYSTEM]

    @property
    def grounding_calls(self) -> int:
        return sum(1 for r in self.requests if r.images)


@pytest.fixture
def session_for(asset, tracks, boxes):  # noqa: F811
    def make(model: Model) -> EditSession:
        return EditSession(model, asset=asset, tracks=tracks, boxes=boxes, frame_loader=FRAME)

    return make


def current_plan_section(prompt: str) -> str:
    match = re.search(r"## 현재 계획.*?\n(.*?)(?:\n## |\Z)", prompt, re.S)
    return match.group(1) if match else ""


class TestSaying:
    def test_clear_instruction_is_ready_to_approve(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        reply = session.say("빨간 옷 입은 사람 모자이크")
        assert reply.status == "plan"
        assert reply.can_approve
        assert session.plan.operations[0].selector.include == ["person_02"]
        assert "이 대상이 맞나요" in reply.text
        assert session.events[-1].llm_calls == 2

    def test_out_of_scope_request_asks_back(self, session_for):
        session = session_for(Model({"operations": [], "clarification": "자막은 지원하지 않아요."}))
        reply = session.say("자막 넣어줘")
        assert (reply.status, reply.text) == ("clarify", "자막은 지원하지 않아요.")
        assert session.plan is None

    def test_ui_state_is_part_of_the_turn(self, session_for):
        model = Model(mosaic("빨간 옷"))
        session_for(model).say("여기서부터", ui=UIContext(playhead=3.0))
        assert "재생 위치: 3.0s" in model.planner_prompts[0]


class TestChoosing:
    def test_open_choice_blocks_approval(self, session_for):
        session = session_for(Model(mosaic("모자 쓴 사람")))
        reply = session.say("모자 쓴 사람 모자이크")
        assert not reply.can_approve
        with pytest.raises(ApprovalBlocked, match="골라 주세요"):
            session.approve()

    def test_picking_a_candidate_settles_it(self, session_for):
        session = session_for(Model(mosaic("모자 쓴 사람")))
        session.say("모자 쓴 사람 모자이크")
        reply = session.choose(0, ["person_03"])
        assert reply.can_approve
        assert session.approve().operations[0].selector.include == ["person_03"]

    def test_clicking_a_box_answers_the_open_choice(self, session_for):
        session = session_for(Model(mosaic("모자 쓴 사람")))
        session.say("모자 쓴 사람 모자이크")
        session.toggle(0, "person_01")
        assert session.plan.operations[0].selector.include == ["person_01"]
        assert session.confirmation.blocking_items == []

    def test_choice_survives_a_spoken_correction(self, session_for):
        """The guess must not be laundered into a settled id by the planner."""
        model = Model(mosaic("모자 쓴 사람"), mosaic("모자 쓴 사람", end=5.0))
        session = session_for(model)
        session.say("모자 쓴 사람 모자이크")
        reply = session.say("5초까지만")

        shown = current_plan_section(model.planner_prompts[1])
        assert "모자 쓴 사람" in shown
        assert "person_01" not in shown and "person_03" not in shown
        assert not reply.can_approve  # still waiting for the user
        assert session.plan.operations[0].time_range.end == 5.0

    def test_regrounding_a_known_phrase_costs_no_call(self, session_for):
        model = Model(mosaic("모자 쓴 사람"), mosaic("모자 쓴 사람", end=5.0))
        session = session_for(model)
        session.say("모자 쓴 사람 모자이크")
        session.say("5초까지만")
        assert model.grounding_calls == 1
        assert session.events[-1].llm_calls == 1

    def test_unknown_track_is_rejected(self, session_for):
        session = session_for(Model(mosaic("모자 쓴 사람")))
        session.say("모자 쓴 사람 모자이크")
        with pytest.raises(ValueError, match="영상에 없는"):
            session.choose(0, ["person_99"])


class TestUnresolved:
    def test_nobody_found_is_never_approved_as_everyone(self, session_for):
        session = session_for(Model(mosaic("유령")))
        reply = session.say("유령 모자이크")
        assert not reply.can_approve
        with pytest.raises(ApprovalBlocked):
            session.approve()

    def test_choosing_replaces_the_marker(self, session_for):
        session = session_for(Model(mosaic("유령")))
        session.say("유령 모자이크")
        session.choose(0, ["person_02"])
        assert session.approve().operations[0].selector.include == ["person_02"]


class TestClicking:
    def test_click_spares_one_from_a_whole_kind_edit(self, session_for):
        session = session_for(Model(mosaic()))
        session.say("사람 전부 모자이크")
        session.toggle(0, "person_02")
        assert session.plan.operations[0].selector.exclude == ["person_02"]
        session.toggle(0, "person_02")
        assert session.plan.operations[0].selector.exclude == []

    def test_click_adds_and_drops_explicit_targets(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        session.toggle(0, "person_01")
        assert session.plan.operations[0].selector.include == ["person_02", "person_01"]
        session.toggle(0, "person_02")
        assert session.plan.operations[0].selector.include == ["person_01"]

    def test_last_target_cannot_be_dropped(self, session_for):
        """Dropping it would leave include empty -- which means everyone."""
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        with pytest.raises(ValueError, match="마지막 대상"):
            session.toggle(0, "person_02")

    def test_time_range_drag(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        session.set_time_range(0, 2.0, 50.0)
        tr = session.plan.operations[0].time_range
        assert (tr.start, tr.end) == (2.0, 10.0)
        with pytest.raises(ValueError):
            session.set_time_range(0, 6.0, 3.0)

    def test_remove_reindexes(self, session_for):
        two = {"operations": mosaic("모자 쓴 사람")["operations"] + mosaic("빨간 옷")["operations"]}
        session = session_for(Model(two))
        session.say("둘 다")
        session.remove(0)
        assert len(session.plan.operations) == 1
        assert session.plan.operations[0].selector.include == ["person_02"]
        assert session.approve()  # the open choice left with its operation

    def test_clicks_cost_no_calls_and_reach_the_history(self, session_for):
        model = Model(mosaic("빨간 옷"), mosaic("빨간 옷"))
        session = session_for(model)
        session.say("빨간 옷 모자이크")
        session.set_time_range(0, 1.0, 4.0)
        assert len(model.requests) == 2
        session.say("좋아")
        assert "[화면 조작] time_range" in model.planner_prompts[1]

    def test_any_edit_withdraws_approval(self, session_for):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        session.approve()
        session.set_time_range(0, 1.0, 4.0)
        assert session.approved is None


class TestApproving:
    def test_approval_names_the_track(self, session_for):
        model = Model(mosaic("빨간 옷"), mosaic("person_02", end=3.0))
        session = session_for(model)
        session.say("빨간 옷 모자이크")
        session.approve()
        session.say("그 사람 3초까지만")
        assert "이름: 빨간 옷" in model.planner_prompts[1]

    def test_render_requires_approval(self, session_for, tmp_path):
        session = session_for(Model(mosaic("빨간 옷")))
        session.say("빨간 옷 모자이크")
        with pytest.raises(ApprovalBlocked):
            session.render(tmp_path / "out.mp4")

    def test_nothing_to_approve(self, session_for):
        with pytest.raises(ApprovalBlocked):
            session_for(Model()).approve()


class TestFailures:
    def test_model_failure_keeps_the_plan(self, session_for):
        model = Model(mosaic("빨간 옷"))
        session = session_for(model)
        session.say("빨간 옷 모자이크")
        before = session.plan

        model.failure = LLMUnavailable("down")
        reply = session.say("블러로 바꿔")
        assert reply.status == "error"
        assert session.plan == before
        assert reply.confirmation is session.confirmation

    def test_quota_says_when_it_resets(self, session_for):
        model = Model()
        model.failure = LLMQuotaExhausted("quota", resets_in=20 * 3600)
        reply = session_for(model).say("모자이크")
        assert "20시간" in reply.text
