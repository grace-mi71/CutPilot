# 0001 — 초기 기술 스택 및 범위 결정

- **상태:** 확정
- **날짜:** 2026-09-15
- **적용 범위:** Week 1–12 전체

계획서(`docs/Project_plan.md`)를 구현으로 옮기기 전에 확정한 사항과 그 근거를
기록한다. "왜 이걸 안 썼지?"를 나중에 다시 고민하지 않기 위한 문서다.

---

## 1. 개발 환경

| 항목 | 결정 | 근거 |
|---|---|---|
| 프로젝트명 | **CutPilot** | 레포·발표 브랜딩 통일 (계획서 원제 EditAgent) |
| 패키지 매니저 | **uv** | torch 커스텀 인덱스 처리가 pip보다 명확 |
| Python | **3.10** | 기존 설치 유지. CV 스택 호환 문제 없음 |
| 영상 처리 | **FFmpeg + OpenCV** | 프록시·스프라이트·최종 렌더 모두 커버 |

### ⚠️ RTX 5060 (Blackwell, sm_120) 주의

로컬 GPU는 Blackwell 세대다. CUDA 12.8 미만 빌드의 PyTorch를 설치하면
`no kernel image is available for execution on the device`로 GPU가 잡히지 않는다.

- torch는 **CUDA 12.8 이상 인덱스를 명시**해 설치한다
- 설치 직후 `torch.cuda.get_device_capability()`가 `(12, 0)`인지 검증한다
- 모든 추론 경로에 `--device cpu` 폴백을 둔다

---

## 2. 학습은 하지 않는다

**Week 1–10 전체에 학습이 0회다.** 전부 pretrained 또는 학습 파라미터가 없는
알고리즘이다.

| 구성 요소 | 학습 필요 | 비고 |
|---|---|---|
| Person / Face / Plate detection | ❌ | pretrained 그대로 |
| Tracking (ByteTrack) | ❌ | IoU + Kalman **알고리즘**. 파라미터 없음 |
| Grounding | ❌ | zero-shot |
| Auto Reframe / Blur | ❌ | 순수 연산 |
| Planner | ❌ | API 호출 |
| 한국 번호판 detector | ⚠️ 유일한 후보 | **파이프라인 완성 후로 유보** |

**따라서 Colab / 클라우드 학습 환경 세팅도 유보한다.** 번호판 fine-tuning을
실제로 시작할 때 `[train]` 의존성 그룹과 가중치 동기화 경로(Drive 또는 HF Hub)를
추가한다.

---

## 3. LLM — Gemini API (Google AI Studio)

### 왜 Gemini인가

Gemini 2.5는 **플래닝 · 박스 · 폴리곤 마스크 · conversational segmentation을
한 모델에서** 제공한다. 박스는 `[ymin, xmin, ymax, xmax]` 0–1000 정규화로 반환된다.
별도 grounding 모델을 붙일 이유가 약해진다.

### 왜 Vertex AI(GCP)가 아닌가

`google-genai` SDK가 두 백엔드를 **같은 코드베이스로** 지원한다. 전환 비용이
클라이언트 생성 한 줄 수준이므로 지금 고민할 가치가 없다.

```python
client = genai.Client(api_key=...)                               # AI Studio
client = genai.Client(vertexai=True, project=..., location=...)  # Vertex
```

Vertex가 주는 것(SLA, VPC-SC, 컴플라이언스, data residency)은 이 프로젝트에서
쓰지 않는다. 게다가 Vertex는 **결제 계정이 필수**이고, 2026-07부터 비(非)글로벌
리전 엔드포인트는 약 10% 비싸다.

### ⚠️ 무료 티어 데이터 사용 정책

**무료 티어 프롬프트는 구글 제품 개선에 사용될 수 있다. 유료 티어와 Vertex는
사용되지 않는다.** 프라이버시 블러가 핵심 기능인 프로젝트라 짚고 간다.

| 구간 | 티어 | 이유 |
|---|---|---|
| Week 1–10 개발 | 무료 | 직접 촬영 영상만 사용 |
| Week 11 평가 | **유료 전환(소액)** | 레이트리밋 회피 + 데이터 미사용 + Pro 접근 |
| 발표 | — | "실서비스라면 Vertex 또는 온프렘" 명시 |

### 재현성

평가 수치가 흔들리지 않도록 **모델 버전을 핀 고정하고 응답을 캐싱**한다.
(`cutpilot/core/`)

---

## 4. 채택하지 않은 모델

### Grounding DINO — 제외

phrase grounding 전용이라 계획서 3.1이 내세운 복합 의도를 처리하지 못한다.

```
"처음부터 등장한"   temporal reference    ← 불가
"다른 사람"         relational reasoning  ← 불가
"왼쪽 여자"         spatial reasoning     ← 취약
"검은 셔츠 남자"     phrase grounding      ← 가능
```

네 줄 중 한 줄만 처리하고 나머지는 어차피 LLM 몫이다. 로컬 폴백이 필요하다면
GDINO보다 **소형 VLM(Moondream 3급)** 이 상위 호환이다 — 자연어를 그대로 받는다.

### SAM2 — 제외 (조건부 재검토)

SAM2가 주는 두 가지의 가치가 갈린다.

| | 필요? | 이유 |
|---|---|---|
| 픽셀 마스크 (가리기용) | ❌ | **박스 블러가 오히려 안전.** 윤곽만 가리면 턱선·귀·머리카락이 남아 재식별 위험 |
| 픽셀 마스크 (보여주기용) | ✅ | 겹친 인물 구분에 필요 — **단, keyframe 몇 장뿐** |
| video memory 전파 | ⚠️ | Risk 2(ID switch) 보험. 3순위 |

확인용 마스크는 Gemini가 폴리곤으로 반환하므로 충분하다.

**재검토 조건:** 블러 자체를 마스크 기반으로 바꾸기로 하면 전 프레임 마스크가
필요해지고, 그때 SAM2 video propagation이 실제 후보가 된다.

### Tracking 안정화 순서

```
1. ByteTrack                 ← 여기서 시작 (공짜, 학습 없음)
   ↓ ID switch가 실제로 발생하면
2. BoT-SORT + ReID embedding ← 가볍고 ID switch를 정조준
   ↓ 그래도 부족하면
3. SAM2 video propagation    ← 8GB에서 부담되는 마지막 카드
```

### 로컬 VLM — Week 11로 유보

에이전트(Planner)를 붙여야 하므로 로컬 선택지는 사실상 지워진다.

| | Grounder | Planner (에이전트) |
|---|---|---|
| 출력 | 박스 몇 개 | **스키마를 지키는 JSON** |
| 턴 | 1턴 | **멀티턴, 이전 plan 유지·수정** |
| 실패 시 | IoU 매칭이 흡수 | **파이프라인 정지** |

8B급 로컬 모델의 고질적 약점이 멀티턴 structured output 준수다. 프로젝트의 핵심
기여(Week 7–8)에 그 리스크를 걸 수 없다.

또한 sm_120용 양자화 스택(`bitsandbytes`, AWQ/GPTQ, `flash-attn`, vLLM) 휠 지원
여부가 불확실하다. 여기서 며칠이 날아가면 Week 7–8이 밀린다.

**Week 11 평가에서 Grounding Accuracy baseline으로만 추가한다.** 안 쓴 게 아니라
비교하고 뺀 것이 되어, 설계 근거가 측정치로 남는다.

---

## 5. 최종 스택

| 역할 | 선택 | 위치 |
|---|---|---|
| Planner | Gemini API (structured output) | `agent/` |
| Grounder | Gemini API → **IoU로 YOLO 박스에 매칭** | `vision/` |
| Person / Face / Plate detection | YOLO (pretrained) | `vision/` |
| Tracking | ByteTrack | `vision/` |
| Reframe / Blur / Render | OpenCV + FFmpeg | `editing/` |
| 프록시 / 스프라이트 / 메타데이터 | FFmpeg | `media/` |

### Grounder는 detector가 아니라 selector다

Gemini의 박스는 정밀하지 않다. 그래서 **선택만 맡기고 좌표는 YOLO가 책임진다.**

```
YOLO      → person_01, person_02, person_03   (정밀한 박스)
Gemini    → "검은 셔츠 남자는 여기쯤"            (대략적인 박스)
IoU 매칭  → person_02                          ← 정밀도 복원
ByteTrack → person_02를 전 프레임 추적
```

---

## 6. 에이전트 구조 — 선언적 Plan, tool calling 아님

계획서 6장(tool calling)과 7장(structured output)이 충돌한다.
**Feature 3(Editable Edit Plan)은 7장 방식이어야만 성립한다.**

tool calling은 호출 즉시 실행된다. 그런데 서비스 기여(3.2)는 *"렌더링 전에 계획을
보여주고 대화로 고친다"* 이다. 실행돼버리면 보여줄 계획이 없다.

```
자연어 + 대화 히스토리
      │  Gemini structured output
      ▼
  EditPlan JSON        ← 선언적. 타임라인에 렌더. 사용자가 수정
      │  사용자 승인
      ▼
  로컬 Executor        ← 여기서 비로소 tool 호출로 변환
      ▼
  track / reframe / blur / render
```

**LLM은 tool을 직접 호출하지 않는다.** 검증·재현·롤백이 쉬워지고, EditPlan이
그대로 평가 대상(12.1 Instruction Understanding)이 된다.

### ⚠️ 스키마는 얕게

Gemini structured output은 OpenAPI 부분집합만 지원한다. Pydantic 중첩 모델이
생성하는 `$ref` / `$defs`에서 걸릴 수 있으므로 **EditPlan 중첩을 얕게 유지**한다.

---

## 7. 타임라인 UX — 계획서 확장

계획서 3.1은 temporal reference를 기여로 내세우면서 10.1 UI 목업에는 **시간축이
없다.** Object Overlay는 "화면 어디"만 보여주고 "언제부터 언제까지"는 못 보여준다.
타임라인이 그 자리를 채운다.

### NLE 타임라인이 아니다 — 범위 방어선

| | CapCut / Clipchamp | CutPilot |
|---|---|---|
| 정체 | 편집 트랙 | **객체 · 작업 트랙** |
| 조작 | 자르기·옮기기·트랜지션 | **읽기 + 확인 + 구간 지정** |
| 편집 주체 | 사용자가 직접 | **대화로 지시, 타임라인은 결과를 비춤** |

계획서 14장의 *"Premiere 수준 Timeline Editor 제외"* 는 이 구분으로 유지된다.
**"CapCut처럼 보이되, CapCut처럼 조작하지는 않는다."**

오히려 이게 슬로건과 맞는다 — 타임라인을 직접 만지는 게 아니라 대화가 만든 결과를
보는 곳으로 두면 *Intent-Based* 가 UI로 증명된다.

### 트랙 구성

```
🎞 ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮      썸네일 스트립

person_02  ████████░░░░░░░░████████    ← 분석 결과 (Track)
person_03       ████████████████
plate_01             ████████
─────────────────────────────────
BLUR p_03        ▓▓▓▓▓▓▓▓▓▓            ← Edit Plan 시각화
REFRAME 9:16 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

위층은 "영상에 뭐가 있나", 아래층은 "내가 뭘 걸었나"다. **아래층이 곧 EditPlan
JSON의 시각화**이므로 텍스트 패널보다 설득력이 높다.

`Track`을 프레임 단위로 소비하고 버리지 말고 **presence segment로 보존**하면
하이라이트가 추가 연산 없이 나온다. 같은 구조가 temporal reference 해석
("처음부터 등장한")과 구간 교집합 계산에도 쓰인다.

> 사용자가 "10–25초"라 했는데 person_02가 12–18초에 없으면 실제 적용 구간은
> `[10,12] ∪ [18,25]`. 타임라인에 그대로 보여주면 즉시 이해된다.

### 더블체크는 축이 두 개

```
공간축  프레임 위 마스크/박스 오버레이   →  "이 사람"
시간축  타임라인 구간 하이라이트         →  "이 구간"
```

둘 다 있어야 확정된다. 겹친 장면에서 박스만 보면 모호하고, 타임라인만 보면 누군지
모른다. 계획서 Risk 1 대응이 "후보 인물 표시" 한 줄인데, 실제로는 이것이 서비스
신뢰의 전부다.

**프리뷰에서 인물을 클릭 → 챗에 `@person_02` 참조 삽입**이 가장 강력한 대응이다.
말로 지칭하는 대신 클릭하면 grounding 실패 자체를 우회한다. Gemini가 반환한 폴리곤
좌표로 히트테스트하면 되므로 난이도도 낮다.

### 마스크 전송은 폴리곤 JSON

PNG 마스크는 무겁고 클릭 판정이 안 된다. 폴리곤 좌표는 가볍고, 캔버스로 그리고,
히트테스트까지 된다. Gemini가 이미 폴리곤으로 준다.

### 업로드 파이프라인은 3개를 뱉는다

```
원본 ├─→ ① 프록시 영상 (480p)     프리뷰·스크럽
     ├─→ ② 썸네일 스프라이트       타임라인 필름스트립
     └─→ ③ 메타데이터             duration, fps, 해상도
```

**프록시가 없으면 UX가 무너진다.** 원본 1080p를 브라우저에서 스크럽하면 버벅인다.
그리고 이것이 계획서 **Risk 4 대응("low-res preview → final render")과 같은
얘기**다. 하나로 두 문제가 풀린다.

```bash
ffmpeg -i in.mp4 -vf scale=-2:480 -c:v libx264 -preset veryfast proxy.mp4
ffmpeg -i in.mp4 -vf "fps=1,scale=160:-1,tile=20x1" sprite.jpg
```

### 범위 — 읽기 전용부터

```
[필수]  구간 하이라이트 · 클릭 시 해당 시점 점프 · 재생 헤드 동기화
        ← 여기까지로 더블체크 목적은 100% 달성
[선택]  타임라인 줌 · 드래그로 구간 조정 · 단축키
```

드래그로 구간을 조정하는 순간 "대화로 편집한다"는 컨셉과도 충돌한다.

---

## 8. 스키마에 미치는 영향

계획서 7장 EditPlan에는 **시간 필드가 없다.** 구간 편집을 하려면 지금 넣어야
한다. 나중에 넣으면 스키마·executor·UI·평가를 전부 고쳐야 한다.

```jsonc
// 계획서 원안 — 항상 영상 전체
{ "type": "FACE_BLUR", "exclude": ["person_02"] }

// 확정안
{
  "type": "PERSON_BLUR",
  "target": "person_02",
  "time_range": { "start": 10.0, "end": 25.0 },  // null = 영상 시작/끝
  "mode": "box"                                   // box | mosaic
}
```

시간 표현은 LLM이 다양하게 뱉는다(`"처음 30초"`, `"끝까지"`, `"저 사람 나오는
동안"`). `null`을 영상 시작/끝으로 두고, `"나오는 동안"`은 해당 Track의 presence
segment로 해석한다.

| 스키마 | 필드 |
|---|---|
| `VideoAsset` | `proxy_url`, `sprite_url`, `duration`, `fps`, `width`, `height` |
| `Track` | `id`, `label`, `segments[(start, end)]`, `boxes` |
| `EditPlan` | operations + 각 operation의 `time_range`, `mode` |

---

## 9. 프론트엔드 — Week 9–10 유지, 단 계약은 지금

| 시점 | 내용 |
|---|---|
| Week 1–2 | **API 응답 스키마 확정** (위 3종) |
| Week 5–6 | 얇은 프로토타입 — 정적 HTML + 캔버스 타임라인 하나 |
| Week 9–10 | 본 구현 |

Week 5–6의 반나절짜리 프로토타입이 "타임라인 렌더링이 생각보다 어렵다"를 4주 먼저
알려준다.

UI가 없는 Week 1–8 동안은 **디버그 렌더(박스 오버레이 mp4 + track 타임라인 콘솔
출력)** 가 유일한 눈이다. 스모크 테스트에 포함한다.

---

## 10. 우선순위에 대한 의견 (미확정)

Auto Reframe은 Premiere·CapCut·Descript에 이미 있다. 시각적 임팩트는 크지만
차별점은 아니다. 반면 **"대화로 구간 지정 → 시각적 더블체크 → 승인 → 실행"** 은
계획서 3.2의 실제 구현체이고 차별점이다.

Week 9–10에 시간이 부족해지면 Auto Reframe 고도화(smoothing 품질 등)를 줄이고
타임라인 UX에 투자하는 쪽이 발표에서 유리하다고 본다. **확정 사항은 아니다.**

---

## 범위 제외 (계획서 14장 + 추가)

계획서 14장 목록을 그대로 유지하고 다음을 명시한다.

- **Object Removal / inpainting** — 가리기는 블러·모자이크로만 한다
- **NLE 스타일 클립 편집** — 자르기·붙이기·트랜지션·다중 트랙
- **모델 학습** — 번호판 detector fine-tuning은 파이프라인 완성 후 별건
