"""State the interface contributes to a turn.

"In the selected range, blur the man behind" only means something because the
user has a range selected. The playhead and selection are part of the
instruction, so they travel with every message rather than living only in the
browser.
"""

from __future__ import annotations

from pydantic import BaseModel

from cutpilot.schemas.plan import TimeRange


class UIContext(BaseModel):
    playhead: float | None = None
    """Seconds. What "here" and "from now on" refer to."""

    selection: TimeRange | None = None
    """The range highlighted on the timeline, if any."""
