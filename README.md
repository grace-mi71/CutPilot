# CutPilot

**Conversational AI Agent for Intent-Based Video Editing**

> From Tool-Based Video Editing to Intent-Based Video Editing.

Ajou University · 2026-2 Media Project (졸업작품)

---

## What it does

Describe the edit you want in plain language. CutPilot turns it into a
structured, reviewable **Edit Plan**, shows you exactly who and what it selected,
and only then renders.

```
User   "검은 옷 입은 남자 중심으로 쇼츠로 만들고, 10초부터 25초까지
        다른 사람 얼굴이랑 번호판은 가려줘."

Agent   EDIT PLAN
        Reframe   9:16, follow person_02          00:00 ─ 00:30
        Blur      faces except person_02          00:10 ─ 00:25
        Blur      license plates                  00:10 ─ 00:25

User   "왼쪽 여자도 남겨줘."     ← plan updates, nothing rendered yet
User   "이대로 편집해."          ← now it renders
```

The timeline highlights each selected object's presence so you can confirm the
agent picked the right person *before* spending a render.

## Core features

1. **Person-Centered Auto Reframe** — 16:9 → 9:16 tracking a chosen subject
2. **Face / License Plate Privacy Blur** — detect, track, blur; per-time-range
3. **Conversational Edit Plan** — declarative plan you revise by talking

## Architecture

The LLM plans; it never touches pixels and never calls editing tools directly.

```
natural language ──▶ Agent (Gemini) ──▶ EditPlan (JSON)
                                            │
                                    user reviews & edits
                                            │
                                            ▼
                                     local Executor
                                            │
                         ┌──────────────────┼──────────────────┐
                         ▼                  ▼                  ▼
                  detect / track      auto reframe       blur / mosaic
                         └──────────────────┼──────────────────┘
                                            ▼
                                   FFmpeg ──▶ output video
```

## Layout

```
backend/cutpilot/
  api/        FastAPI routers
  core/       settings, logging, LLM response cache
  schemas/    VideoAsset · Track · EditPlan  (backend ↔ UI contract)
  media/      proxy transcode, thumbnail sprite, probe
  vision/     detection · tracking · grounding
  agent/      planner + LLM provider abstraction
  editing/    reframe · blur · render
  eval/       evaluation: tasks, scorer, simulated user
backend/eval/ labelled task sets (frozen tracks + ground truth)
docs/         API contract, evaluation, decision records
frontend/     Next.js editor: preview with clickable boxes, chat, work list, spec
```

## Status

Working end to end: upload → plan (one sentence or several steps) → the agent
asks only what it cannot decide → final spec → approve → render.
See `docs/api.md` for the API and `docs/evaluation.md` for the first results.

## Run

```bash
# backend (http://localhost:8000)
cd backend
uv run uvicorn cutpilot.api.app:app --port 8000

# frontend (http://localhost:3000; ?mode=text hides the boxes for the user study)
cd frontend
npm install
npm run dev
```

`backend/.env` holds `CUTPILOT_GEMINI_API_KEY`. Without a key the backend runs an
offline keyword planner. Spend is capped in code by `CUTPILOT_LLM_BUDGET_USD`
(estimated from token counts, kept in `outputs/.llm_cache/spend.json`).

```bash
# evaluation
cd backend
uv run python scripts/run_eval.py eval/crowd --max-usd 2
```

## Prerequisites

- Python 3.10
- [FFmpeg](https://ffmpeg.org/) on `PATH`
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU recommended (CPU fallback supported)
