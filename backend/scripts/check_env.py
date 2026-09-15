"""Verify the local toolchain before anything else is built on top of it.

Run:  uv run python scripts/check_env.py

Checks, in order of how badly they break the project:
    1. GPU   -- an RTX 50-series card needs a CUDA 12.8+ torch build. An older
                build imports fine and only fails later, deep in a render, with
                "no kernel image is available for execution on the device".
                So we actually run a kernel rather than trusting is_available().
    2. FFmpeg -- proxy transcode, thumbnail sprite and final render all shell
                out to it.
    3. Imports -- the vision and agent stacks resolve.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"

# Blackwell (RTX 50xx) is sm_120; kernels for it ship in CUDA 12.8 and later.
MIN_CAPABILITY_FOR_BLACKWELL = (12, 0)


def check_gpu() -> tuple[str, list[str]]:
    lines: list[str] = []
    try:
        import torch
    except ImportError as exc:
        return FAIL, [f"torch not importable: {exc}"]

    lines.append(f"torch        {torch.__version__}")
    lines.append(f"cuda build   {torch.version.cuda}")

    if not torch.cuda.is_available():
        lines.append("cuda         not available -- CPU only")
        lines.append("             fine for writing code, too slow for video")
        return WARN, lines

    cap = torch.cuda.get_device_capability()
    props = torch.cuda.get_device_properties(0)
    lines.append(f"device       {props.name}")
    lines.append(f"capability   sm_{cap[0]}{cap[1]}")
    lines.append(f"vram         {props.total_memory / 1024**3:.1f} GiB")
    lines.append(f"arch list    {', '.join(torch.cuda.get_arch_list())}")

    # The real test: is_available() is True even when no kernel matches this
    # architecture. Only launching one proves the build is usable.
    try:
        x = torch.randn(2048, 2048, device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
        lines.append("matmul       ran on GPU")
    except RuntimeError as exc:
        lines.append(f"matmul       FAILED: {exc}")
        if cap >= MIN_CAPABILITY_FOR_BLACKWELL:
            lines.append(
                "             this card needs a CUDA 12.8+ torch build; "
                "reinstall from the cu128 or cu130 index"
            )
        return FAIL, lines

    return OK, lines


def check_ffmpeg() -> tuple[str, list[str]]:
    lines: list[str] = []
    status = OK
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        if path is None:
            lines.append(f"{tool:<12} not on PATH")
            status = FAIL
            continue
        version = subprocess.run(
            [tool, "-version"], capture_output=True, text=True, check=False
        ).stdout.splitlines()[0]
        lines.append(f"{tool:<12} {version.split(' Copyright')[0]}")
    if status is FAIL:
        lines.append("             open a new terminal if it was just installed")
    return status, lines


def check_imports() -> tuple[str, list[str]]:
    modules = {
        "cv2": "opencv",
        "ultralytics": "YOLO detection",
        "supervision": "ByteTrack + annotators",
        "google.genai": "Gemini client",
        "fastapi": "API layer",
        "pydantic": "schemas",
    }
    lines: list[str] = []
    status = OK
    for module, role in modules.items():
        try:
            mod = __import__(module, fromlist=["__version__"])
            version = getattr(mod, "__version__", "?")
            lines.append(f"{module:<16} {version:<12} {role}")
        except ImportError as exc:
            lines.append(f"{module:<16} {'MISSING':<12} {role}  ({exc})")
            status = FAIL
    return status, lines


def main() -> int:
    checks = (
        ("GPU", check_gpu),
        ("FFmpeg", check_ffmpeg),
        ("Imports", check_imports),
    )

    worst = OK
    for name, check in checks:
        status, lines = check()
        print(f"\n{status}  {name}")
        for line in lines:
            print(f"        {line}")
        if status is FAIL or (status is WARN and worst is OK):
            worst = status

    print()
    if worst is FAIL:
        print("Environment is not ready.")
        return 1
    if worst is WARN:
        print("Environment usable with caveats above.")
        return 0
    print("Environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
