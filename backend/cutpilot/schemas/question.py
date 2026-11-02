"""The agent's side of the conversation: one question at a time, then a spec.

The user hands over a plan once. The agent works through it and stops only
where it cannot go on without them -- an open "which one?", a target it could
not find, a guess worth checking, a gap or crop the render would hide. Each of
those becomes a Question, asked in plan order. When none is left, what the
agent will do is written out as a Spec, and only an approved spec is rendered.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cutpilot.schemas.plan import TimeRange

QuestionKind = Literal["choose", "missing", "confirm", "track_gap", "out_of_frame"]
"""choose        several candidates, or nobody matched -- pick one (blocking)
   missing       the operation selects nothing -- rephrase or drop it (blocking)
   confirm       the agent picked one -- is that right?
   track_gap     a target vanishes mid-range -- cover the gap or leave it
   out_of_frame  the reframe crops a target away -- keep or drop the edit"""

QUESTION_ORDER: dict[str, int] = {
    "missing": 0,
    "choose": 1,
    "confirm": 2,
    "track_gap": 3,
    "out_of_frame": 4,
}


class QuestionOption(BaseModel):
    """A one-click answer. Clicking a box or typing are always possible too."""

    id: str
    label: str


class Question(BaseModel):
    id: str
    """Stable while the thing it asks about is unchanged; send it back to answer."""

    kind: QuestionKind
    operation_index: int | None = None
    text: str
    blocking: bool
    """True: approval is impossible until this is settled -- it cannot be waved
    through with an option, only answered."""

    options: list[QuestionOption] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    """Track ids to highlight. For choose: pick among these. For confirm: the guess."""

    track_id: str | None = None
    time_range: TimeRange | None = None
    seek_t: float | None = None
    """Where to put the preview so the user can see what is being asked."""


class SpecStep(BaseModel):
    index: int
    op: str
    label: str
    targets: list[str] = Field(default_factory=list)
    """Track ids, with what the user called them where known: "person_02 (빨간 옷)"."""
    excluded: list[str] = Field(default_factory=list)
    time: str


class Spec(BaseModel):
    """What will be rendered, exactly, once approved."""

    steps: list[SpecStep] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    """How each question was answered, in order -- the spec's audit trail."""
    text: str = ""
