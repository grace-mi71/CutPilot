"""The uploaded video and everything derived from it at ingest time."""

from __future__ import annotations

from pydantic import BaseModel


class VideoAsset(BaseModel):
    """An upload plus its two derived artifacts.

    The proxy and the sprite sheet exist because the timeline has to stay
    responsive while the user is still deciding what to edit. Scrubbing 1080p in
    a browser stutters, so preview and seeking run on the proxy and only the
    approved plan is rendered at source resolution. That also defers the one
    expensive step until after the user has agreed to it.
    """

    id: str
    source_path: str
    proxy_path: str
    sprite_path: str

    duration: float
    fps: float
    width: int
    height: int

    sprite_interval: float = 1.0
    """Seconds between thumbnails on the sprite sheet."""

    sprite_columns: int = 20

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def sprite_count(self) -> int:
        if self.sprite_interval <= 0:
            return 0
        return int(self.duration / self.sprite_interval) + 1

    def clamp_time(self, t: float) -> float:
        return min(max(t, 0.0), self.duration)
