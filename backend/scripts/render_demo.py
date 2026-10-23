"""Render a hand-written EditPlan over a clip -- the executor, end to end.

    uv run python scripts/render_demo.py data/samples/<clip>.mp4
    uv run python scripts/render_demo.py <clip>.mp4 --reframe 16:9

No LLM is involved. The plan is built in code exactly as the agent would emit
it after grounding, which is the point: it proves the back half of the pipeline
works before the front half exists, and it is the same path the agent's output
will take later.

Rendering runs on the proxy by default, so the result arrives in seconds and is
produced by the identical code that renders the final video at source
resolution -- which is what makes a preview worth trusting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cutpilot.core.config import get_settings
from cutpilot.editing import execute
from cutpilot.media import ingest as ingest_module
from cutpilot.schemas.plan import EditPlan, Operation, Selector, TimeRange
from cutpilot.vision import detect
from cutpilot.vision.resolve import resolve_selector


def _section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def build_demo_plan(
    video_id: str,
    tracks: list,
    duration: float,
    *,
    op: str,
    start: float | None,
    end: float | None,
    spare_first: bool,
    reframe_ratio: str | None,
) -> EditPlan:
    """The plan the agent would produce for a sentence like:

    "mosaic everyone's face from 5 to 15 seconds, except that person"
    """
    operations: list[Operation] = []
    subject = tracks[0].id if tracks else None

    if reframe_ratio and subject:
        operations.append(
            Operation(
                op="REFRAME",
                selector=Selector(kind="person", include=[subject]),
                time_range=TimeRange.whole(),
                aspect_ratio=reframe_ratio,
            )
        )

    operations.append(
        Operation(
            op=op,
            selector=Selector(
                kind="person",
                exclude=[subject] if (spare_first and subject) else [],
            ),
            time_range=TimeRange(start=start, end=end),
        )
    )
    return EditPlan(video_id=video_id, operations=operations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--op", choices=["MOSAIC", "BLUR"], default="MOSAIC")
    parser.add_argument("--start", type=float, default=5.0, help="null for video start")
    parser.add_argument("--end", type=float, default=15.0, help="null for video end")
    parser.add_argument("--spare-first", action="store_true", help="exclude person_01")
    parser.add_argument("--reframe", default=None, metavar="RATIO", help='e.g. "16:9"')
    parser.add_argument("--full-res", action="store_true", help="render the source, not the proxy")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    device = args.device or settings.resolved_device()

    if not args.video.exists():
        print(f"not found: {args.video}", file=sys.stderr)
        return 1

    _section("1. Ingest")
    asset = ingest_module.ingest(
        args.video,
        settings.output_dir,
        proxy_height=settings.proxy_height,
        sprite_interval=settings.sprite_interval,
        sprite_thumb_width=settings.sprite_thumb_width,
        sprite_columns=settings.sprite_columns,
    )
    print(f"  {asset.width}x{asset.height}  {asset.fps:.2f}fps  {asset.duration:.2f}s")

    _section(f"2. Detect + track  (device={device})")
    started = time.perf_counter()
    result = detect.detect_and_track(
        Path(asset.proxy_path),
        model_path=str(settings.weights_dir / settings.person_model),
        fps=asset.fps,
        kind="person",
        device=device,
        confidence=settings.detect_confidence,
    )
    print(f"  {len(result.tracks)} tracks in {time.perf_counter() - started:.1f}s")
    if not result.tracks:
        print("  FAIL  nothing tracked", file=sys.stderr)
        return 1

    _section("3. Edit plan")
    plan = build_demo_plan(
        asset.id,
        result.tracks,
        asset.duration,
        op=args.op,
        start=args.start,
        end=args.end,
        spare_first=args.spare_first,
        reframe_ratio=args.reframe,
    )
    for index, op in enumerate(plan.operations):
        targets = resolve_selector(op.selector, result.tracks)
        scope = "all" if op.selector.is_whole_kind else f"{op.selector.include}"
        spared = f" except {op.selector.exclude}" if op.selector.exclude else ""
        print(
            f"  [{index}] {op.op:<8} {scope}{spared}"
            f"  {op.time_range.describe(asset.duration)}"
            f"  -> {len(targets)} tracks"
        )

    _section("4. Render")
    source = Path(asset.source_path) if args.full_res else Path(asset.proxy_path)
    suffix = "full" if args.full_res else "proxy"
    target = settings.output_dir / asset.id / f"render_{suffix}.mp4"

    started = time.perf_counter()
    render = execute(
        plan,
        asset,
        result.tracks,
        result.boxes,
        target,
        source=source,
        blur_strength=settings.default_blur_strength,
        mosaic_strength=settings.default_mosaic_strength,
        smoothing_window=settings.reframe_smoothing_window,
    )
    elapsed = time.perf_counter() - started

    print(f"  source    {source.name}")
    print(f"  output    {render.aspect}  {render.frames} frames")
    print(f"  regions   {render.regions_drawn} drawn")
    if render.regions_cropped_away:
        print(
            f"  warning   {render.regions_cropped_away} landed outside the reframe "
            "crop -- the user should have been told before approving"
        )
    print(f"  {target}")
    print(f"  {target.stat().st_size / 1024**2:.1f} MB  {elapsed:.1f}s")

    print("\n\033[1mRender OK.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
