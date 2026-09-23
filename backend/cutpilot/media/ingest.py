"""Ingest pipeline -- proxy transcode, thumbnail sprite, probe metadata.

Every upload produces three artifacts so the timeline UI can render without
touching the original file:
    1. proxy video   (low-res, fast scrubbing)
    2. sprite sheet  (timeline filmstrip)
    3. metadata      (duration, fps, resolution)

The proxy is not only a UI convenience. Preview and confirmation run on it while
only the approved plan is rendered at source resolution, which is what keeps the
one expensive step behind the user's approval.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cutpilot.schemas.video import VideoAsset


class FFmpegNotFound(RuntimeError):
    pass


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FFmpegNotFound(
            f"{tool} is not on PATH. Install it (winget install Gyan.FFmpeg) "
            "and open a new terminal so the PATH change takes effect."
        )
    return path


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{args[0]} failed:\n{result.stderr.strip()[-2000:]}")
    return result


def _parse_rate(value: str) -> float:
    """r_frame_rate arrives as a fraction such as "25/1"."""
    if match := re.fullmatch(r"(\d+)/(\d+)", value.strip()):
        num, den = int(match[1]), int(match[2])
        return num / den if den else 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(source: Path) -> VideoInfo:
    ffprobe = _require("ffprobe")
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ProbeError(f"no video stream in {source}")

    stream = streams[0]
    duration = float(payload.get("format", {}).get("duration", 0.0))
    fps = _parse_rate(stream.get("r_frame_rate", "0/1"))
    if duration <= 0 or fps <= 0:
        raise ProbeError(f"unusable duration/fps in {source}: {duration=} {fps=}")

    return VideoInfo(
        duration=duration,
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def make_proxy(source: Path, target: Path, *, height: int = 480) -> Path:
    """Low-resolution copy for preview and scrubbing.

    Width is forced even (-2) because H.264 rejects odd dimensions, which bites
    on portrait sources where the scaled width lands on an odd number.
    """
    ffmpeg = _require("ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale=-2:{height}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-an",
            str(target),
        ]
    )
    return target


def make_sprite(
    source: Path,
    target: Path,
    *,
    duration: float,
    interval: float = 1.0,
    thumb_width: int = 160,
    columns: int = 20,
) -> tuple[int, int]:
    """Filmstrip for the timeline. Returns (thumbnail count, row count).

    One sheet rather than N requests: a 3-minute clip at one thumbnail per
    second is 180 files, and the timeline wants all of them at once.
    """
    ffmpeg = _require("ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)

    count = max(1, int(duration / interval) + 1)
    rows = math.ceil(count / columns)
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"fps=1/{interval},scale={thumb_width}:-2,tile={columns}x{rows}",
            "-frames:v",
            "1",
            str(target),
        ]
    )
    return count, rows


def ingest(
    source: Path,
    output_root: Path,
    *,
    proxy_height: int = 480,
    sprite_interval: float = 1.0,
    sprite_thumb_width: int = 160,
    sprite_columns: int = 20,
    video_id: str | None = None,
) -> VideoAsset:
    """Run the full ingest and return the asset the rest of the pipeline uses."""
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    info = probe(source)
    vid = video_id or re.sub(r"[^A-Za-z0-9_.-]", "_", source.stem)
    work = output_root / vid

    proxy = make_proxy(source, work / "proxy.mp4", height=proxy_height)
    sprite = work / "sprite.jpg"
    make_sprite(
        source,
        sprite,
        duration=info.duration,
        interval=sprite_interval,
        thumb_width=sprite_thumb_width,
        columns=sprite_columns,
    )

    return VideoAsset(
        id=vid,
        source_path=str(source),
        proxy_path=str(proxy),
        sprite_path=str(sprite),
        duration=info.duration,
        fps=info.fps,
        width=info.width,
        height=info.height,
        sprite_interval=sprite_interval,
        sprite_columns=sprite_columns,
    )
