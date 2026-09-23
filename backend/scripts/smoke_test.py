"""End-to-end smoke test: a video in, a tracked overlay out.

    uv run python scripts/smoke_test.py data/samples/<clip>.mp4

Covers the week 1-2 plumbing in one pass -- ingest, detect, track, and both
debug views -- so a break anywhere shows up before the agent is layered on top.
Nothing here calls an LLM.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cutpilot.core.config import get_settings
from cutpilot.media import ingest as ingest_module
from cutpilot.schemas.plan import Selector
from cutpilot.vision import debug, detect
from cutpilot.vision.resolve import resolve_selector


def _section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="source clip")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: settings)")
    parser.add_argument("--max-frames", type=int, default=None, help="cap for a quick pass")
    parser.add_argument("--tracker", default=detect.TRACKER_BYTETRACK)
    parser.add_argument("--no-overlay", action="store_true", help="skip the debug render")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    device = args.device or settings.resolved_device()

    if not args.video.exists():
        print(f"not found: {args.video}", file=sys.stderr)
        return 1

    # -- 1. ingest ---------------------------------------------------------
    _section("1. Ingest")
    started = time.perf_counter()
    try:
        asset = ingest_module.ingest(
            args.video,
            settings.output_dir,
            proxy_height=settings.proxy_height,
            sprite_interval=settings.sprite_interval,
            sprite_thumb_width=settings.sprite_thumb_width,
            sprite_columns=settings.sprite_columns,
        )
    except ingest_module.FFmpegNotFound as exc:
        print(f"  FAIL  {exc}", file=sys.stderr)
        return 1

    print(f"  id        {asset.id}")
    print(f"  source    {asset.width}x{asset.height}  {asset.fps:.2f}fps  {asset.duration:.2f}s")
    print(f"  proxy     {Path(asset.proxy_path).name}")
    print(f"  sprite    {Path(asset.sprite_path).name}  ({asset.sprite_count} thumbs)")
    print(f"  {time.perf_counter() - started:.1f}s")

    if asset.width > asset.height:
        print("  note      landscape source -- Auto Reframe has something to convert")
    else:
        print("  note      portrait source -- Auto Reframe (16:9 -> 9:16) is untestable here")

    # -- 2. detect + track -------------------------------------------------
    _section(f"2. Detect + track  (device={device}, tracker={args.tracker})")
    started = time.perf_counter()
    result = detect.detect_and_track(
        Path(asset.proxy_path),
        model_path=str(settings.weights_dir / settings.person_model),
        fps=asset.fps,
        kind="person",
        device=device,
        confidence=settings.detect_confidence,
        tracker=args.tracker,
        max_frames=args.max_frames,
    )
    elapsed = time.perf_counter() - started
    fps_processed = result.frames_processed / elapsed if elapsed else 0.0
    print(f"  frames    {result.frames_processed}")
    print(f"  detections{result.detections_total:>6}")
    print(f"  tracks    {len(result.tracks)}")
    if result.dropped_tracks:
        print(f"  dropped   {result.dropped_tracks} (too brief to be editable objects)")
    print(f"  {elapsed:.1f}s  ({fps_processed:.1f} fps)")

    if not result.tracks:
        print("  FAIL  nothing tracked", file=sys.stderr)
        return 1

    # -- 3. timeline -------------------------------------------------------
    _section("3. Track timeline")
    print(debug.format_timeline(result.tracks, asset.duration))

    gapped = [t for t in result.tracks if len(t.segments) > 1]
    print(f"\n  {len(gapped)} of {len(result.tracks)} tracks have gaps")
    if gapped:
        worst = max(gapped, key=lambda t: len(t.segments))
        print(f"  most fragmented: {worst.id} ({len(worst.segments)} segments)")
        print("  each gap is a question for the user: did they leave, or was the")
        print("  tracker lost? Only they know, so it becomes a track_gap warning.")

    # -- 4. selector ------------------------------------------------------
    _section("4. Selector resolution")
    everyone = resolve_selector(Selector(kind="person"), result.tracks)
    print(f"  kind=person, include=[]          -> {len(everyone)} tracks")
    if len(result.tracks) >= 2:
        spared = result.tracks[0].id
        rest = resolve_selector(Selector(kind="person", exclude=[spared]), result.tracks)
        print(f"  exclude=[{spared}]{'':<10}-> {len(rest)} tracks (spared {spared})")

    # -- 5. overlay --------------------------------------------------------
    if args.no_overlay:
        _section("5. Debug overlay  (skipped)")
    else:
        _section("5. Debug overlay")
        started = time.perf_counter()
        target = settings.output_dir / asset.id / "debug_overlay.mp4"
        debug.render_overlay(
            Path(asset.proxy_path),
            target,
            result.tracks,
            result.boxes,
            fps=asset.fps,
            max_frames=args.max_frames,
        )
        size_mb = target.stat().st_size / 1024**2
        print(f"  {target}")
        print(f"  {size_mb:.1f} MB  {time.perf_counter() - started:.1f}s")

    print("\n\033[1mPipeline OK.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
