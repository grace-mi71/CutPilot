"""Run an evaluation set against the live model and write a report.

    uv run python scripts/run_eval.py eval/crowd --max-usd 2

Each task costs one planner call plus one grounding call per described person.
Answers are cached (outputs/.llm_cache), so a rerun of unchanged tasks is free
and reproduces the same numbers. Spend is checked between tasks and the run
stops at --max-usd, on top of the global cap in CUTPILOT_LLM_BUDGET_USD.

Writes outputs/eval/<set>-<timestamp>.json (every record) and .md (the tables).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cutpilot.agent.grounder import video_frame_loader
from cutpilot.agent.llm import LLMError, provider_from_settings
from cutpilot.agent.session import EditSession
from cutpilot.core.config import get_settings
from cutpilot.eval.run import run_task, summarize
from cutpilot.eval.tasks import TaskSet
from cutpilot.schemas.video import VideoAsset
from cutpilot.vision.detect import TrackingResult
from cutpilot.vision.heads import with_head_regions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("set_dir", type=Path)
    parser.add_argument("--max-usd", type=float, default=2.0)
    parser.add_argument("--only", nargs="*", help="task ids to run")
    args = parser.parse_args()

    settings = get_settings()
    tasks = TaskSet.load(args.set_dir / "tasks.json")
    analysis = TrackingResult.load(args.set_dir / "tracks.json")
    if settings.estimate_faces:
        analysis = with_head_regions(analysis)
    asset = VideoAsset.model_validate_json((args.set_dir / "asset.json").read_text("utf-8"))
    source = settings.data_dir / tasks.source
    if not source.exists():
        print(f"missing video: {source}", file=sys.stderr)
        return 2
    asset = asset.model_copy(update={"source_path": str(source), "proxy_path": str(source)})

    provider = provider_from_settings(settings)
    summary = getattr(provider, "summary", None)

    def spent() -> float:
        return float(summary()["usd"]) if summary else 0.0

    loader = video_frame_loader(source, asset.fps)

    def make_session() -> EditSession:
        return EditSession(
            provider, asset=asset, tracks=analysis.tracks, boxes=analysis.boxes,
            frame_loader=loader, planner_thinking=settings.planner_thinking_level,
            grounding_thinking=settings.grounding_thinking_level,
        )  # fmt: skip

    start = spent()
    records = []
    for task in tasks.tasks:
        if args.only and task.id not in args.only:
            continue
        if spent() - start >= args.max_usd:
            print(f"stopping: spent ${spent() - start:.4f} of ${args.max_usd}")
            break
        try:
            pair = run_task(
                task, make_session, tracks=analysis.tracks, duration=asset.duration, spent=spent
            )
        except LLMError as exc:
            print(f"{task.id}: model error {type(exc).__name__}: {exc}")
            continue
        records += pair
        auto, neg = pair
        print(
            f"{task.id:6} auto={'O' if auto.success else 'X'} "
            f"neg={'O' if neg.success else 'X'}  "
            f"missed {auto.missed_s:5.1f}s -> {neg.missed_s:5.1f}s  "
            f"q={neg.questions} fix={neg.corrections} calls={neg.llm_calls} "
            f"${neg.usd:.4f} {neg.seconds:.1f}s"
        )

    rows = summarize(records)
    out_dir = settings.output_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"{tasks.name}-{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps(
            {"set": tasks.name, "labeled_by": tasks.labeled_by, "summary": rows,
             "records": [r.as_dict() for r in records], "usd": round(spent() - start, 5)},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )  # fmt: skip
    base.with_suffix(".md").write_text(report(tasks.name, rows, records), encoding="utf-8")
    print(f"\nspent ${spent() - start:.4f}  ->  {base}.md")
    return 0


def report(name: str, rows: list[dict], records) -> str:
    lines = [
        f"# Evaluation: {name}",
        "",
        "| level | condition | tasks | success | missed s | extra s | questions/task "
        "| needless questions | clicks/task | calls/task |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['level']} | {r['condition']} | {r['tasks']} | {r['success']:.0%} | "
            f"{r['missed_s']} | {r['extra_s']} | {r['questions']} | {r['needless_q']} | "
            f"{r['corrections']} | "
            f"{r['llm_calls']} |"
        )
    lines += [
        "",
        "## Per task",
        "",
        "| task | auto | negotiated | missed s (auto → neg) | questions | clicks | could not fix |",
        "|---|---|---|---|---|---|---|",
    ]
    by_task: dict[str, dict] = {}
    for r in records:
        by_task.setdefault(r.task, {})[r.condition] = r
    for task_id, pair in by_task.items():
        a, n = pair["auto"], pair["negotiated"]
        lines.append(
            f"| {task_id} | {'O' if a.success else 'X'} | {'O' if n.success else 'X'} | "
            f"{a.missed_s} → {n.missed_s} | {n.questions} | {n.corrections} | "
            f"{'; '.join(n.unfixable) or '-'} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
