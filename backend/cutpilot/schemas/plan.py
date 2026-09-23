"""The Negotiable Edit Plan.

The plan is what the user and the agent negotiate over, so its shape is a design
decision rather than a serialization detail. It is split along the lines a user
actually corrects things: "no, the person behind" touches the selector, "cover
his whole body" touches the region, "only up to 50 seconds" touches the time
range. Each correction leaves the other fields standing, so nobody has to repeat
a whole instruction to fix one word of it.

Two variants exist:

    EditPlanDraft   what the LLM returns -- targets are still phrases
    EditPlan        after grounding -- targets are track ids

Keeping them apart means the grounding step has a typed input and output, and
the LLM never has to invent an identifier.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from cutpilot.schemas.track import ObjectKind

OpKind = Literal["BLUR", "MOSAIC", "REFRAME"]

# MVP ships "bbox" only: the detected box of whatever the selector picked.
# Every demo instruction resolves through `kind` instead -- "just the faces" is
# kind="face", not region="face" on a person. The other values are reserved so
# that adding sub-object regions later is a one-line change here and one branch
# in the executor, rather than a new field threaded through the LLM prompt, the
# confirmation UI and every stored plan.
Region = Literal["bbox", "face", "body"]


class TimeRange(BaseModel):
    """Where an operation applies. None means "to the edge of the video"."""

    start: float | None = None
    end: float | None = None

    @classmethod
    def whole(cls) -> TimeRange:
        return cls()

    def resolve(self, duration: float) -> tuple[float, float]:
        start = 0.0 if self.start is None else max(0.0, self.start)
        end = duration if self.end is None else min(duration, self.end)
        return start, max(start, end)

    def intersect(self, start: float, end: float, duration: float) -> tuple[float, float] | None:
        lo, hi = self.resolve(duration)
        lo, hi = max(lo, start), min(hi, end)
        return (lo, hi) if hi > lo else None

    def describe(self, duration: float) -> str:
        start, end = self.resolve(duration)
        if start == 0.0 and end >= duration:
            return "전체 구간"
        return f"{start:.1f}s ─ {end:.1f}s"


class Selector(BaseModel):
    """Which objects an operation applies to.

    One structure covers the three ways people point at things:

        "cover every face"          include empty      -> all of `kind`
        "cover just that one"       include=[id]
        "cover everyone but him"    exclude=[id]

    Ids may be given for a different kind than `kind` -- grounding resolves a
    person, but the operation may target that person's face. Translation happens
    in vision.resolve, not here.
    """

    kind: ObjectKind
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    @property
    def is_whole_kind(self) -> bool:
        return not self.include


class Operation(BaseModel):
    """One edit. Every op carries the same fields; unused ones stay None.

    A discriminated union would be tidier, but Gemini's structured output
    supports only a subset of OpenAPI and handles `anyOf` poorly. A single flat
    shape is worth the nulls.
    """

    op: OpKind
    selector: Selector
    region: Region = "bbox"
    time_range: TimeRange = Field(default_factory=TimeRange)

    aspect_ratio: str | None = None  # REFRAME only, e.g. "9:16"
    strength: int | None = None  # BLUR / MOSAIC only

    @model_validator(mode="after")
    def _check_op_fields(self) -> Operation:
        if self.op == "REFRAME":
            if self.aspect_ratio is None:
                raise ValueError("REFRAME requires aspect_ratio")
            if len(self.selector.include) != 1:
                raise ValueError("REFRAME follows exactly one subject")
        return self

    @property
    def is_privacy_op(self) -> bool:
        return self.op in ("BLUR", "MOSAIC")


class EditPlan(BaseModel):
    """A grounded, executable plan. This is what the user approves."""

    video_id: str
    operations: list[Operation] = Field(default_factory=list)

    @property
    def reframe(self) -> Operation | None:
        """At most one reframe applies -- it reshapes the whole output."""
        return next((op for op in self.operations if op.op == "REFRAME"), None)

    @property
    def privacy_ops(self) -> list[Operation]:
        return [op for op in self.operations if op.is_privacy_op]

    def referenced_track_ids(self) -> set[str]:
        ids: set[str] = set()
        for op in self.operations:
            ids.update(op.selector.include)
            ids.update(op.selector.exclude)
        return ids


# --------------------------------------------------------------------------
# Draft -- the LLM's output, before grounding
# --------------------------------------------------------------------------


class SelectorDraft(BaseModel):
    """Targets as the user said them: "the man in the black cap", "the woman on
    the left". Grounding turns these into track ids."""

    kind: ObjectKind
    include_phrases: list[str] = Field(default_factory=list)
    exclude_phrases: list[str] = Field(default_factory=list)


class OperationDraft(BaseModel):
    op: OpKind
    selector: SelectorDraft
    time_range: TimeRange = Field(default_factory=TimeRange)
    aspect_ratio: str | None = None
    strength: int | None = None


class EditPlanDraft(BaseModel):
    operations: list[OperationDraft] = Field(default_factory=list)
