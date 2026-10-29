"""Where uploads and conversations live between requests.

Videos are analysed once and written to disk (asset.json, tracks.json under
outputs/<id>/), so a restarted server still knows them. Sessions are held in
memory only: a conversation is cheap to restart and nothing about it is worth
a database yet.

Each session carries a lock. FastAPI runs plain handlers on a thread pool, and
two clicks arriving together would otherwise interleave edits to one plan.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cutpilot.agent.grounder import video_frame_loader
from cutpilot.agent.llm import LLMProvider
from cutpilot.agent.session import EditSession
from cutpilot.api.models import RenderOut
from cutpilot.core.config import Settings
from cutpilot.schemas.video import VideoAsset
from cutpilot.vision.detect import TrackingResult

Analyzer = Callable[[Path, str], tuple[VideoAsset, TrackingResult]]
"""(uploaded file, video id) -> the asset and its tracks."""


def default_analyzer(settings: Settings) -> Analyzer:
    """Ingest plus person tracking -- the same steps scripts/render_demo runs."""

    def analyze(source: Path, video_id: str) -> tuple[VideoAsset, TrackingResult]:
        from cutpilot.media import ingest as ingest_module
        from cutpilot.vision import detect

        asset = ingest_module.ingest(
            source,
            settings.output_dir,
            proxy_height=settings.proxy_height,
            sprite_interval=settings.sprite_interval,
            sprite_thumb_width=settings.sprite_thumb_width,
            sprite_columns=settings.sprite_columns,
            video_id=video_id,
        )
        # Tracked on the proxy: identical normalised boxes, a fraction of the time.
        analysis = detect.detect_and_track(
            Path(asset.proxy_path),
            model_path=str(settings.weights_dir / settings.person_model),
            fps=asset.fps,
            kind="person",
            device=settings.resolved_device(),
            confidence=settings.detect_confidence,
        )
        return asset, analysis

    return analyze


def new_video_id(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename).stem)[:40] or "video"
    return f"{stem}-{uuid.uuid4().hex[:6]}"


@dataclass
class VideoEntry:
    asset: VideoAsset
    analysis: TrackingResult


class Library:
    def __init__(self, output_dir: Path, analyze: Analyzer) -> None:
        self.output_dir = output_dir
        self.analyze = analyze
        self._videos: dict[str, VideoEntry] = {}

    def _dir(self, video_id: str) -> Path:
        return self.output_dir / video_id

    def add(self, source: Path, video_id: str) -> VideoEntry:
        asset, analysis = self.analyze(source, video_id)
        work = self._dir(asset.id)
        analysis.save(work / "tracks.json")
        (work / "asset.json").write_text(asset.model_dump_json(indent=1), encoding="utf-8")
        entry = VideoEntry(asset=asset, analysis=analysis)
        self._videos[asset.id] = entry
        return entry

    def get(self, video_id: str) -> VideoEntry | None:
        if video_id in self._videos:
            return self._videos[video_id]
        work = self._dir(video_id)
        if "/" in video_id or "\\" in video_id or not (work / "asset.json").exists():
            return None
        asset = VideoAsset.model_validate(
            json.loads((work / "asset.json").read_text(encoding="utf-8"))
        )
        entry = VideoEntry(asset=asset, analysis=TrackingResult.load(work / "tracks.json"))
        self._videos[video_id] = entry
        return entry


@dataclass
class SessionEntry:
    id: str
    video_id: str
    session: EditSession
    lock: threading.Lock = field(default_factory=threading.Lock)
    render: RenderOut | None = None
    """The latest render of the approved plan."""


class Sessions:
    def __init__(self, provider_factory: Callable[[], LLMProvider]) -> None:
        self._provider_factory = provider_factory
        self._provider: LLMProvider | None = None
        self._sessions: dict[str, SessionEntry] = {}

    @property
    def provider(self) -> LLMProvider:
        # Built on first use: a server with no key still serves uploads.
        if self._provider is None:
            self._provider = self._provider_factory()
        return self._provider

    def create(self, video: VideoEntry) -> SessionEntry:
        asset = video.asset
        source = Path(asset.source_path)
        frames = source if source.exists() else Path(asset.proxy_path)
        session = EditSession(
            self.provider,
            asset=asset,
            tracks=video.analysis.tracks,
            boxes=video.analysis.boxes,
            frame_loader=video_frame_loader(frames, asset.fps),
        )
        entry = SessionEntry(id=uuid.uuid4().hex[:12], video_id=asset.id, session=session)
        self._sessions[entry.id] = entry
        return entry

    def get(self, session_id: str) -> SessionEntry | None:
        return self._sessions.get(session_id)
