"""Writing rendered frames out as a playable video.

Frames go straight into ffmpeg over a pipe rather than through
cv2.VideoWriter. Two reasons: one H.264 encode instead of writing an
intermediate file and re-encoding it, and the source's audio can be muxed in
during the same pass. An editor that silently drops the audio track is not
finished.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class EncoderError(RuntimeError):
    pass


class FrameWriter:
    """Pipe BGR frames to ffmpeg. Use as a context manager."""

    def __init__(
        self,
        target: Path,
        *,
        width: int,
        height: int,
        fps: float,
        audio_from: Path | None = None,
        crf: int = 20,
        preset: str = "veryfast",
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise EncoderError("ffmpeg is not on PATH")

        target.parent.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.width = width
        self.height = height
        self.frames_written = 0

        args = [
            ffmpeg,
            "-v", "error",
            "-y",
            # raw frames on stdin
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps}",
            "-i", "-",
        ]  # fmt: skip
        if audio_from is not None:
            args += ["-i", str(audio_from)]

        args += [
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",  # without this many players show nothing
        ]  # fmt: skip

        if audio_from is not None:
            # "?" keeps a silent source from failing the whole render, and
            # -shortest stops at the video when the audio runs longer.
            args += ["-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]

        args.append(str(target))
        self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise EncoderError("writer is closed")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise EncoderError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, "
                f"writer expects {self.width}x{self.height}"
            )
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise EncoderError(self._drain() or "ffmpeg closed the pipe") from exc
        self.frames_written += 1

    def _drain(self) -> str:
        if self._process.stderr is None:
            return ""
        return self._process.stderr.read().decode(errors="replace").strip()

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        code = self._process.wait()
        message = self._drain()
        if code != 0:
            raise EncoderError(f"ffmpeg exited {code}: {message[-2000:]}")

    def __enter__(self) -> FrameWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._process.kill()
            self._process.wait()
            return
        self.close()


def has_audio(source: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return False
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(source),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())
