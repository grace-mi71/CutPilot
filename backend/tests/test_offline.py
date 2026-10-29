"""The offline keyword planner: enough understanding to drive the interface."""

from __future__ import annotations

import json

from cutpilot.agent.llm.offline import CLARIFY, plan_from_keywords
from cutpilot.agent.planner import build_prompt
from cutpilot.schemas import EditPlan, Operation, Selector, TimeRange
from cutpilot.schemas.session import UIContext


def plan(text: str, current: EditPlan | None = None, ui: UIContext | None = None) -> dict:
    prompt = build_prompt(text, tracks=[], duration=20.0, ui=ui, current_plan=current)
    return plan_from_keywords(prompt)


def only_op(text: str, **kwargs) -> dict:
    return plan(text, **kwargs)["operations"][0]


def test_described_target_becomes_a_phrase():
    op = only_op("가방 멘 사람 모자이크")
    assert op["op"] == "MOSAIC"
    assert op["selector"]["include_phrases"] == ["가방 멘 사람"]


def test_kind_and_times():
    op = only_op("5초부터 10초까지 얼굴 블러")
    assert (op["op"], op["selector"]["kind"]) == ("BLUR", "face")
    assert op["time_range"] == {"start": 5.0, "end": 10.0}


def test_until_only():
    assert only_op("person_02 모자이크 7초까지")["time_range"] == {"start": None, "end": 7.0}


def test_ids_and_exclusion():
    op = only_op("person_01 빼고 전부 모자이크")
    assert op["selector"]["include_phrases"] == []
    assert op["selector"]["exclude_phrases"] == ["person_01"]


def test_selected_range():
    ui = UIContext(selection=TimeRange(start=2.0, end=4.0))
    assert only_op("선택한 구간 블러", ui=ui)["time_range"] == {"start": 2.0, "end": 4.0}


def test_reframe():
    op = only_op("person_03 중심으로 세로로")
    assert (op["op"], op["aspect_ratio"]) == ("REFRAME", "9:16")
    assert op["selector"]["include_phrases"] == ["person_03"]


def test_correction_edits_the_current_plan():
    current = EditPlan(
        video_id="v",
        operations=[Operation(op="MOSAIC", selector=Selector(kind="face", include=["person_02"]))],
    )
    op = only_op("블러로 바꾸고 3초부터 6초까지", current=current)
    assert op["op"] == "BLUR"
    assert op["selector"] == {
        "kind": "face", "include_phrases": ["person_02"], "exclude_phrases": [],
    }  # fmt: skip
    assert op["time_range"] == {"start": 3.0, "end": 6.0}


def test_unknown_request_asks_back():
    result = plan("배경음악 넣어줘")
    assert result == {"operations": [], "clarification": CLARIFY}
    json.dumps(result)
