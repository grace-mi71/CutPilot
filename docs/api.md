# CutPilot API

프론트엔드가 호출하는 백엔드 HTTP API. 응답 필드의 정확한 스키마는 서버 실행 후
`http://localhost:8000/docs` (자동 생성 문서)에서 확인할 수 있다.

## 실행

```bash
cd backend
uv run uvicorn cutpilot.api.app:app --reload --port 8000
```

| 환경 변수 | 값 | 설명 |
|---|---|---|
| `CUTPILOT_LLM_PROVIDER` | `auto` (기본) | API 키가 있으면 Gemini, 없으면 오프라인 |
| | `offline` | 키워드 규칙으로 동작하는 가짜 플래너. **할당량 소모 없음**, UI 개발용 |
| | `gemini` | 항상 Gemini |

오프라인 모드는 키워드만 알아듣는다 (`가방 멘 사람 모자이크`, `5초부터 10초까지 얼굴 블러`,
`person_02 빼고 전부 모자이크`, `세로로`). 외모로 지칭한 대상은 항상 "누구인가요?"(choose)로
돌아오므로 후보 선택 UI를 개발하기에 좋다. `GET /sessions/{id}`의 `provider` 값이
`offline-keywords`이면 화면에 "오프라인 모드" 표시를 권장.

CORS는 `http://localhost:3000`이 허용되어 있다 (`CUTPILOT_CORS_ORIGINS`로 변경).

## 흐름

```
POST /videos                업로드 → 분석 (짧은 영상 기준 수십 초, 동기)
POST /sessions              대화 시작
  ┌ POST   .../messages                    채팅 (말로 지시·수정)      → Reply
  │ POST   .../operations/{i}/choose       후보 중 선택               → Reply
  │ POST   .../operations/{i}/toggle       미리보기 박스 클릭         → Reply
  │ PUT    .../operations/{i}/time_range   타임라인 구간 수정         → Reply
  └ DELETE .../operations/{i}              작업 삭제                  → Reply
POST /sessions/{id}/approve  승인 → 렌더링 → 결과 영상 URL
```

`{i}`는 작업 리스트의 행 번호(0부터). 모든 수정 요청은 같은 모양의 **Reply**를 돌려주므로,
화면은 매번 Reply 하나로 다시 그리면 된다.

## 엔드포인트

### `POST /videos` — 업로드
`multipart/form-data`, 필드 이름 `file`. 영상 길이 상한 180초 (초과 시 413).

```jsonc
// 201
{
  "id": "crowd-2edf12",
  "duration": 20.64, "fps": 25.0, "width": 1080, "height": 1920,
  "proxy_url": "/files/crowd-2edf12/proxy.mp4",     // 미리보기는 이걸 재생 (원본보다 가벼움)
  "sprite_url": "/files/crowd-2edf12/sprite.jpg",   // 타임라인 썸네일 스트립
  "sprite_interval": 1.0, "sprite_columns": 20,
  "tracks": [                                       // 타임라인 레인: 대상별 등장 구간
    { "id": "person_01", "kind": "person", "parent_id": null, "display_name": null,
      "segments": [{ "start": 0.0, "end": 20.64 }] }
  ]
}
```
URL은 서버 기준 상대 경로 → `http://localhost:8000` + url.

### `GET /videos/{id}` — 위와 같은 응답 (새로고침 시)

### `GET /videos/{id}/boxes?t=3.2` — 그 순간 화면에 있는 모든 대상의 박스
재생 위치에서 클릭 가능한 박스를 그릴 때 사용.
```json
{ "t": 3.2, "boxes": [{ "track_id": "person_01", "kind": "person",
                        "box": { "x": 0.12, "y": 0.30, "w": 0.08, "h": 0.25 } }] }
```
**좌표는 모두 0~1 정규화** (영상 너비·높이에 곱해서 사용).

### `POST /sessions` — `{ "video_id": "crowd-2edf12" }` → Session

### `POST /sessions/{id}/messages`
```jsonc
{ "text": "가방 멘 사람 모자이크",
  "ui": { "playhead": 2.0,                         // 현재 재생 위치 (초)
          "selection": { "start": 1.0, "end": 5.0 } } }  // 타임라인 선택 구간, 없으면 null
```
`ui`는 "여기서부터", "선택한 구간" 같은 말을 해석하는 데 쓰이므로 **매번 같이 보낸다**.

### Reply (모든 대화·수정 요청의 응답)
```jsonc
{
  "status": "plan",            // plan | clarify (되묻기) | error (AI 실패, 계획은 그대로)
  "text": "이렇게 이해했습니다: ...",   // 채팅에 표시할 문장
  "notes": [],
  "can_approve": false,        // 승인 버튼 활성화 여부
  "blockers": ["1번 작업: 대상을 골라 주세요 (후보: person_01, ...)."],
  "confirmation": {
    "keyframe_t": 2.0,         // 박스를 보여줄 프레임: 미리보기를 이 시간으로 이동
    "plan": { "video_id": "...", "operations": [
      { "op": "MOSAIC",                         // MOSAIC | BLUR | REFRAME
        "selector": { "kind": "person",         // person | face | license_plate
                      "include": ["person_01"], // 비어 있으면 그 종류 "전체"
                      "exclude": [] },
        "region": "bbox",
        "time_range": { "start": 1.0, "end": 8.0 },   // null = 영상 처음/끝
        "aspect_ratio": null, "strength": null }
    ]},
    "items": [                 // 작업 리스트 한 행 = 같은 operation_index의 항목 3개
      { "operation_index": 0, "field": "op",         "label": "모자이크",       "status": "ok",     "candidates": [] },
      { "operation_index": 0, "field": "selector",   "label": "사람 · 가방 멘 사람", "status": "choose", "candidates": ["person_01", "person_02"] },
      { "operation_index": 0, "field": "time_range", "label": "1.0s ─ 8.0s",   "status": "ok",     "candidates": [] }
    ],
    "overlays": [              // keyframe_t에서 미리보기 위에 그릴 박스
      { "track_id": "person_01", "box": { "x": 0.1, "y": 0.3, "w": 0.08, "h": 0.25 },
        "role": "candidate", "label": null, "operation_index": 0 }
    ],
    "warnings": [
      { "kind": "track_gap", "message": "person_03이(가) 4.0~6.0s 동안 보이지 않습니다. ...",
        "operation_index": 0, "track_id": "person_03", "time_range": { "start": 4.0, "end": 6.0 } }
    ]
  }
}
```

화면 표시 규칙:

| 값 | 의미 | 권장 표시 |
|---|---|---|
| item `status: ok` | 확실함 | 그냥 표시 |
| item `status: confirm` | AI가 하나 골랐음, 맞는지 확인 | "이 대상이 맞나요?" |
| item `status: choose` | 후보가 여럿 / 못 찾음 | 후보 박스 강조 + 선택 요청, **승인 불가** |
| overlay `role: target` | 편집될 대상 | 주 색상 |
| overlay `role: candidate` | 골라야 할 후보 | 점선 / 깜빡임 |
| overlay `role: excluded` | 제외된 대상 | 회색 |
| overlay `role: reframe_subject` | 세로 변환 기준 인물 | 별도 색 |
| warning `track_gap` | 대상이 중간에 사라짐 (가려짐이면 그 구간은 처리 안 됨) | 타임라인에 해당 구간 표시 |
| warning `out_of_frame` | 세로 변환 후 화면 밖이라 효과가 안 보임 | 경고 |
| warning `empty_selection` | 대상 없음 | 경고, **승인 불가** |

### 클릭 수정
```jsonc
POST   /sessions/{id}/operations/0/choose      { "track_ids": ["person_02"] }  // 후보 선택 (아무 대상이나 가능)
POST   /sessions/{id}/operations/0/toggle      { "track_id": "person_04" }     // 박스 클릭: 넣기/빼기
PUT    /sessions/{id}/operations/0/time_range  { "start": 2.0, "end": 9.5 }
DELETE /sessions/{id}/operations/0
```
toggle 규칙: 대상이 "전체"인 작업에서 클릭 → 제외/제외 해제. 대상이 지정된 작업에서 클릭 → 추가/해제.
선택 대기 중이면 클릭이 곧 선택. 마지막 남은 대상은 해제할 수 없음 (400, 대신 작업 삭제).

### `POST /sessions/{id}/approve` — `{ "preview": true }`
`preview: true`는 프록시 해상도로 1초 내외, `false`는 원본 해상도.
```json
{ "plan": { ... }, "render": { "url": "/files/.../renders/05a2_preview.mp4",
  "preview": true, "width": 270, "height": 480, "frames": 516 } }
```
"완료" 탭에서 `render.url`을 재생. 승인 후 수정하면 승인이 풀리므로 다시 승인해야 한다.

### `GET /sessions/{id}` — 새로고침 복구용
`confirmation`, `can_approve`, `blockers`, `approved`, `history` (채팅 기록), `render`, `provider`.

## 오류

| 코드 | 의미 | 응답 |
|---|---|---|
| 400 | 잘못된 클릭·수정 (없는 대상, 잘못된 구간), 영상이 아닌 파일 | `{ "detail": "메시지" }` |
| 404 | 영상·세션·작업 없음 | `{ "detail": "메시지" }` |
| 409 | 승인 거부 | `{ "detail": { "reasons": ["..."] } }` |
| 413 | 영상이 너무 김 | `{ "detail": "메시지" }` |

AI 호출 실패는 HTTP 오류가 아니라 `status: "error"`인 Reply로 온다 (계획은 그대로 유지).

## 현재 한계
- 서버를 재시작하면 세션은 사라진다 (영상 분석 결과는 디스크에 남음).
- 탐지는 사람만 한다. 얼굴·번호판 탐지는 아직 없음 → "얼굴" 요청은 대상 없음 경고.
- 업로드와 분석은 동기 처리 (진행률 없음). 20초 샘플 기준 약 20~35초.
