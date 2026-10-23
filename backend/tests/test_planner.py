"""Intent parsing -- mostly the normalisation that turns a schema-valid answer
into a meaningful one."""

from __future__ import annotations

import pytest

from cutpilot.agent.llm import FakeProvider
from cutpilot.agent.planner import (
    Turn,
    build_prompt,
    is_complement,
    normalize_draft,
    parse_intent,
    track_id_of,
)
from cutpilot.schemas import (
    EditPlan,
    EditPlanDraft,
    Operation,
    OperationDraft,
    Segment,
    Selector,
    SelectorDraft,
    TimeRange,
    Track,
)
from cutpilot.schemas.session import UIContext

DURATION = 30.0


def _draft(*ops: OperationDraft) -> EditPlanDraft:
    return EditPlanDraft(operations=list(ops))


def _mosaic(include=(), exclude=(), kind="face", start=None, end=None) -> OperationDraft:
    return OperationDraft(
        op="MOSAIC",
        selector=SelectorDraft(
            kind=kind, include_phrases=list(include), exclude_phrases=list(exclude)
        ),
        time_range=TimeRange(start=start, end=end),
    )


class TestComplement:
    @pytest.mark.parametrize(
        "phrase", ["다른 사람들", "나머지", "모든 사람", "전부", "다른 사람들의 얼굴", "the others"]
    )
    def test_everyone_words_are_complements(self, phrase):
        assert is_complement(phrase)

    @pytest.mark.parametrize(
        "phrase", ["검은 옷 입은 남자", "왼쪽 사람", "person_03", "모자 쓴 사람"]
    )
    def test_actual_targets_are_not(self, phrase):
        assert not is_complement(phrase)


class TestNormalize:
    def test_drops_the_complement_seen_in_the_first_live_probe(self):
        """The exact answer Gemini gave on 2026-10-23."""
        draft = _draft(_mosaic(include=["다른 사람들"], exclude=["검은 옷 입은 남자"]))
        normalized, notes = normalize_draft(draft, DURATION)
        selector = normalized.operations[0].selector
        assert selector.include_phrases == []
        assert selector.exclude_phrases == ["검은 옷 입은 남자"]
        assert notes and "다른 사람들" in notes[0]

    def test_keeps_real_targets_alongside(self):
        draft = _draft(_mosaic(include=["왼쪽 여자", "나머지"]))
        normalized, _ = normalize_draft(draft, DURATION)
        assert normalized.operations[0].selector.include_phrases == ["왼쪽 여자"]

    def test_click_references_become_plain_ids(self):
        draft = _draft(_mosaic(include=["@person_03", "person_03"]))
        normalized, _ = normalize_draft(draft, DURATION)
        assert normalized.operations[0].selector.include_phrases == ["person_03"]

    def test_times_are_clamped_and_ordered(self):
        draft = _draft(_mosaic(start=40.0, end=-5.0))
        normalized, _ = normalize_draft(draft, DURATION)
        assert normalized.operations[0].time_range == TimeRange(start=0.0, end=DURATION)

    def test_reframe_is_forced_to_people_with_a_default_ratio(self):
        draft = _draft(
            OperationDraft(
                op="REFRAME",
                selector=SelectorDraft(kind="face", include_phrases=["검은 옷 남자"]),
            )
        )
        op = normalize_draft(draft, DURATION)[0].operations[0]
        assert (op.selector.kind, op.aspect_ratio) == ("person", "9:16")

    def test_only_one_reframe_survives(self):
        reframe = OperationDraft(
            op="REFRAME", selector=SelectorDraft(kind="person", include_phrases=["a"])
        )
        normalized, notes = normalize_draft(_draft(reframe, reframe.model_copy()), DURATION)
        assert sum(o.op == "REFRAME" for o in normalized.operations) == 1
        assert notes

    def test_clarification_passes_through(self):
        draft = EditPlanDraft(clarification="어떤 부분을 편집할까요?")
        normalized, _ = normalize_draft(draft, DURATION)
        assert normalized.clarification == "어떤 부분을 편집할까요?"


class TestTrackId:
    def test_recognises_ids(self):
        assert track_id_of("@face_07") == "face_07"
        assert track_id_of("license_plate_01") == "license_plate_01"
        assert track_id_of("검은 옷") is None


TRACKS = [
    Track(id="person_01", kind="person", segments=[Segment(start=0.0, end=12.0)]),
    Track(
        id="person_02",
        kind="person",
        display_name="검은 모자를 쓴 사람",
        segments=[Segment(start=5.0, end=30.0)],
    ),
]


class TestPrompt:
    def test_selection_is_spelled_out(self):
        """'the selected range' only means something if the numbers are there."""
        prompt = build_prompt(
            "선택 구간 가려줘",
            tracks=TRACKS,
            duration=DURATION,
            ui=UIContext(playhead=45.0, selection=TimeRange(start=10.0, end=20.0)),
        )
        assert "선택 구간: 10.0s ~ 20.0s" in prompt

    def test_tracks_carry_presence_and_names(self):
        prompt = build_prompt("x", tracks=TRACKS, duration=DURATION)
        assert "person_02 (person)  등장 5.0–30.0s  이름: 검은 모자를 쓴 사람" in prompt

    def test_current_plan_is_included_for_corrections(self):
        plan = EditPlan(
            video_id="v",
            operations=[
                Operation(op="MOSAIC", selector=Selector(kind="face", include=["face_03"]))
            ],
        )
        prompt = build_prompt("아니 뒷사람", tracks=TRACKS, duration=DURATION, current_plan=plan)
        assert "현재 계획" in prompt and "face_03" in prompt

    def test_history_is_trimmed(self):
        history = [Turn("user", f"turn {i}") for i in range(20)]
        prompt = build_prompt("x", tracks=TRACKS, duration=DURATION, history=history)
        assert "turn 19" in prompt and "turn 0\n" not in prompt


class TestParseIntent:
    def test_round_trip_through_a_fake_model(self):
        fake = FakeProvider(
            queue=[
                {
                    "operations": [
                        {
                            "op": "MOSAIC",
                            "selector": {
                                "kind": "face",
                                "include_phrases": ["다른 사람들"],
                                "exclude_phrases": ["검은 옷 입은 남자"],
                            },
                            "time_range": {"start": 10, "end": 25},
                        }
                    ]
                }
            ]
        )
        result = parse_intent(fake, "검은 옷 남자 빼고 다 가려줘", tracks=TRACKS, duration=DURATION)
        op = result.draft.operations[0]
        assert op.selector.include_phrases == []
        assert op.time_range == TimeRange(start=10.0, end=25.0)
        assert fake.last.system and "다른 사람들" in fake.last.system
        assert fake.last.thinking_level == "low"
