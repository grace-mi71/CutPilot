"""The HTTP surface: upload, look, talk, click, approve.

    uv run uvicorn cutpilot.api.app:app --reload

Everything is synchronous on purpose. Analysis of a short clip takes seconds
and a proxy render about one, so a job queue would be machinery without a
user. The routes are thin: each one finds a session and calls the matching
EditSession method, which is where the rules live.

Errors map to status codes the frontend can branch on:

    400  the click or edit itself is invalid (unknown track, bad range)
    404  no such video, session or operation
    409  approval refused -- body.detail.reasons says what is still open
    413  the upload is longer than the configured cap
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cutpilot.agent.llm import LLMProvider, provider_from_settings
from cutpilot.agent.session import ApprovalBlocked, Reply, approval_blockers
from cutpilot.api.models import (
    Approve,
    ApproveOut,
    BoxesOut,
    Choose,
    CreateSession,
    RenderOut,
    ReplyOut,
    SendMessage,
    SessionOut,
    SetTimeRange,
    Toggle,
    TrackBox,
    TurnOut,
    VideoOut,
)
from cutpilot.api.offline_demo import create_offline_demo_router
from cutpilot.api.store import (
    Analyzer,
    Library,
    SessionEntry,
    Sessions,
    VideoEntry,
    default_analyzer,
    new_video_id,
)
from cutpilot.core.config import Settings, get_settings

FILES_PREFIX = "/files"


def create_app(
    settings: Settings | None = None,
    *,
    analyze: Analyzer | None = None,
    provider_factory: Callable[[], LLMProvider] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    uploads = settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    library = Library(settings.output_dir, analyze or default_analyzer(settings))
    sessions = Sessions(provider_factory or (lambda: provider_from_settings(settings)))

    app = FastAPI(title="CutPilot", version="0.1.0")
    app.include_router(create_offline_demo_router())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount(FILES_PREFIX, StaticFiles(directory=settings.output_dir), name="files")
    app.state.library = library
    app.state.sessions = sessions

    # ------------------------------------------------------------ helpers

    def url_of(path: str | Path) -> str:
        relative = Path(path).resolve().relative_to(settings.output_dir.resolve())
        return f"{FILES_PREFIX}/{relative.as_posix()}"

    def video_out(entry: VideoEntry) -> VideoOut:
        a = entry.asset
        return VideoOut(
            id=a.id, duration=a.duration, fps=a.fps, width=a.width, height=a.height,
            proxy_url=url_of(a.proxy_path), sprite_url=url_of(a.sprite_path),
            sprite_interval=a.sprite_interval, sprite_columns=a.sprite_columns,
            tracks=entry.analysis.tracks,
        )  # fmt: skip

    def find_video(video_id: str) -> VideoEntry:
        entry = library.get(video_id)
        if entry is None:
            raise HTTPException(404, f"video {video_id} not found")
        return entry

    def find_session(session_id: str) -> SessionEntry:
        entry = sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, f"session {session_id} not found")
        return entry

    def reply_out(reply: Reply) -> ReplyOut:
        c = reply.confirmation
        return ReplyOut(
            status=reply.status, text=reply.text, notes=reply.notes, confirmation=c,
            can_approve=reply.can_approve, blockers=approval_blockers(c) if c else [],
        )  # fmt: skip

    def run(session_id: str, action: Callable[[SessionEntry], Reply]) -> ReplyOut:
        entry = find_session(session_id)
        with entry.lock:
            try:
                return reply_out(action(entry))
            except IndexError as exc:
                raise HTTPException(404, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

    # -------------------------------------------------------------- health

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "llm": settings.llm_provider, "has_key": settings.has_llm}

    # -------------------------------------------------------------- videos

    @app.post("/videos", response_model=VideoOut, status_code=201)
    def upload(file: Annotated[UploadFile, File()]) -> VideoOut:
        if file.content_type and not file.content_type.startswith("video/"):
            raise HTTPException(400, "video files only")
        video_id = new_video_id(file.filename or "video.mp4")
        target = uploads / f"{video_id}{Path(file.filename or '.mp4').suffix or '.mp4'}"
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        from cutpilot.media.ingest import probe

        try:
            info = probe(target)
        except Exception as exc:  # noqa: BLE001 - anything ffprobe cannot read
            target.unlink(missing_ok=True)
            raise HTTPException(400, f"not a readable video: {exc}") from exc
        if info.duration > settings.max_video_seconds:
            target.unlink(missing_ok=True)
            raise HTTPException(
                413, f"영상은 {settings.max_video_seconds:.0f}초 이하만 올릴 수 있습니다."
            )
        return video_out(library.add(target, video_id))

    @app.get("/videos/{video_id}", response_model=VideoOut)
    def get_video(video_id: str) -> VideoOut:
        return video_out(find_video(video_id))

    @app.get("/videos/{video_id}/boxes", response_model=BoxesOut)
    def boxes_at(video_id: str, t: float = Query(ge=0)) -> BoxesOut:
        entry = find_video(video_id)
        out = []
        for track in entry.analysis.tracks:
            source = entry.analysis.boxes.get(track.id)
            box = source.at(t) if source and track.present_at(t) else None
            if box is not None:
                out.append(TrackBox(track_id=track.id, kind=track.kind, box=box))
        return BoxesOut(t=t, boxes=out)

    # ------------------------------------------------------------ sessions

    @app.post("/sessions", response_model=SessionOut, status_code=201)
    def create_session(body: CreateSession) -> SessionOut:
        return session_out(sessions.create(find_video(body.video_id)))

    def session_out(entry: SessionEntry) -> SessionOut:
        s = entry.session
        c = s.confirmation
        return SessionOut(
            id=entry.id, video_id=entry.video_id,
            provider=getattr(sessions.provider, "model", None) or type(sessions.provider).__name__,
            confirmation=c,
            can_approve=bool(c) and not approval_blockers(c),
            blockers=approval_blockers(c) if c else [],
            approved=s.approved,
            history=[TurnOut(role=t.role, text=t.text) for t in s.history],
            render=entry.render,
        )  # fmt: skip

    @app.get("/sessions/{session_id}", response_model=SessionOut)
    def get_session(session_id: str) -> SessionOut:
        return session_out(find_session(session_id))

    @app.post("/sessions/{session_id}/messages", response_model=ReplyOut)
    def send(session_id: str, body: SendMessage) -> ReplyOut:
        return run(session_id, lambda e: e.session.say(body.text, ui=body.ui))

    @app.post("/sessions/{session_id}/operations/{index}/choose", response_model=ReplyOut)
    def choose(session_id: str, index: int, body: Choose) -> ReplyOut:
        return run(session_id, lambda e: e.session.choose(index, body.track_ids, side=body.side))

    @app.post("/sessions/{session_id}/operations/{index}/toggle", response_model=ReplyOut)
    def toggle(session_id: str, index: int, body: Toggle) -> ReplyOut:
        return run(session_id, lambda e: e.session.toggle(index, body.track_id))

    @app.put("/sessions/{session_id}/operations/{index}/time_range", response_model=ReplyOut)
    def set_time_range(session_id: str, index: int, body: SetTimeRange) -> ReplyOut:
        return run(session_id, lambda e: e.session.set_time_range(index, body.start, body.end))

    @app.delete("/sessions/{session_id}/operations/{index}", response_model=ReplyOut)
    def remove(session_id: str, index: int) -> ReplyOut:
        return run(session_id, lambda e: e.session.remove(index))

    @app.post("/sessions/{session_id}/approve", response_model=ApproveOut)
    def approve(session_id: str, body: Approve | None = None) -> ApproveOut:
        body = body or Approve()
        entry = find_session(session_id)
        with entry.lock:
            try:
                plan = entry.session.approve()
            except ApprovalBlocked as exc:
                raise HTTPException(409, {"reasons": exc.reasons}) from exc
            kind = "preview" if body.preview else "final"
            target = settings.output_dir / entry.video_id / "renders" / f"{entry.id}_{kind}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            result = entry.session.render(
                target,
                preview=body.preview,
                blur_strength=settings.default_blur_strength,
                mosaic_strength=settings.default_mosaic_strength,
                smoothing_window=settings.reframe_smoothing_window,
            )
            entry.render = RenderOut(
                url=url_of(target), preview=body.preview, width=result.output_width,
                height=result.output_height, frames=result.frames,
            )  # fmt: skip
        return ApproveOut(plan=plan, render=entry.render)

    return app


def __getattr__(name: str):
    # `uvicorn cutpilot.api.app:app` -- built on first access, not at import,
    # so tests can import create_app without touching the real output folder.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
