# EditAgent
## Conversational AI Agent for Intent-Based Video Editing

**프로젝트 유형:** 미디어 프로젝트 / 졸업작품  
**개발 기간:** 약 3개월 (12주)  
**프로젝트 성격:** 서비스 중심 End-to-End AI 프로젝트  
**핵심 키워드:** Conversational Agent, Multimodal AI, Video Editing, Object Grounding, Tracking, Auto Reframe, Privacy Blur

---

# 1. 프로젝트 개요

## 1.1 한 문장 요약

**EditAgent는 사용자가 자연어로 영상 편집 의도를 전달하면, AI Agent가 이를 실행 가능한 편집 계획으로 변환하고 사용자와 대화를 통해 수정한 뒤 실제 영상 편집까지 수행하는 대화형 영상 편집 서비스이다.**

---

## 1.2 문제 정의

기존 영상 편집은 사용자가 직접 편집 도구의 기능과 조작법을 이해해야 한다.

예를 들어 사용자가 다음과 같은 결과를 원한다고 가정한다.

> "검은 셔츠 입은 남자 중심으로 쇼츠로 만들어줘. 다른 사람 얼굴이랑 자동차 번호판은 모두 가려줘."

기존 영상 편집 프로그램에서는 사용자가 직접 다음 작업을 수행해야 한다.

1. 영상에서 대상 인물 탐색
2. 대상 인물의 프레임별 위치 추적
3. 16:9 영상을 9:16으로 재구성
4. 대상 인물이 중앙에 오도록 crop 위치 조정
5. 다른 사람 얼굴 탐지 및 마스킹
6. 차량 번호판 탐지 및 마스킹
7. 블러 효과 적용
8. 전체 영상 렌더링

즉 사용자는 **원하는 결과가 아니라 편집 방법 자체를 알아야 한다.**

EditAgent는 이를 다음과 같이 바꾸는 것을 목표로 한다.

```text
기존 방식

사용자
  ↓
편집 기능 탐색
  ↓
타임라인 조작
  ↓
마스크 / 트래킹
  ↓
렌더링


EditAgent

사용자 의도
  ↓
자연어 명령
  ↓
AI Agent
  ↓
Edit Plan
  ↓
영상 편집 도구 실행
  ↓
결과 영상
```

즉 프로젝트의 핵심 방향은 다음과 같다.

> **From Tool-Based Video Editing to Intent-Based Video Editing**

---

# 2. 프로젝트 목표

EditAgent의 목표는 범용 영상 편집 프로그램을 만드는 것이 아니다.

3개월이라는 제한된 기간 동안 **시각적으로 명확하고 구현 가능성이 높은 핵심 편집 기능 2개와 대화형 Agent 기능 1개**에 집중한다.

최종적으로 사용자는 영상 편집 기술을 몰라도 다음과 같은 대화를 통해 영상을 편집할 수 있어야 한다.

```text
User:
"검은 옷 입은 남자 중심으로 쇼츠로 만들어줘.
다른 사람 얼굴이랑 번호판은 가려줘."

Agent:
"다음과 같이 이해했습니다.

- 메인 인물: 검은 옷을 입은 남성
- 출력 비율: 9:16
- 메인 인물 중심 Auto Reframe
- 메인 인물을 제외한 다른 얼굴 Blur
- 차량 번호판 Blur"

User:
"왼쪽 여자도 남겨줘."

Agent:
"수정했습니다.

Keep:
- 검은 옷 남성
- 왼쪽 여성

Blur:
- 그 외 얼굴
- 차량 번호판"

User:
"이대로 편집해."
```

이후 Agent가 실제 영상 편집을 수행하고 결과 영상을 생성한다.

---

# 3. 핵심 Contribution

## 3.1 Main Contribution

### Conversational Intent-Based Video Editing

본 프로젝트의 핵심 contribution은 새로운 segmentation 모델이나 video foundation model을 제안하는 것이 아니다.

대신 다음 문제를 해결한다.

> **사용자의 복합적인 자연어 편집 의도를 이해하고, 이를 실제 영상 객체 및 편집 operation과 연결하여 실행 가능한 Edit Plan으로 변환하는 대화형 영상 편집 Agent**

예를 들어 다음 명령을 고려한다.

> "처음부터 등장한 검은 셔츠 남자는 남겨두고 다른 사람 얼굴은 전부 가려줘."

이 명령에는 다음과 같은 복합적인 의미가 포함된다.

```text
"처음부터 등장한"
→ temporal reference

"검은 셔츠 남자"
→ visual grounding

"남겨두고"
→ exclusion condition

"다른 사람"
→ relational reasoning

"얼굴"
→ target region

"전부"
→ entire video scope

"가려줘"
→ blur operation
```

EditAgent는 이러한 사용자의 표현을 영상 속 실제 객체와 연결하고 적절한 편집 operation으로 변환한다.

---

## 3.2 Service Contribution

### Editable Edit Plan

Agent가 사용자의 명령을 받자마자 영상을 렌더링하지 않는다.

먼저 사용자의 명령을 구조화된 **Edit Plan**으로 보여준다.

예시:

```text
EDIT PLAN

Output
- Format: Shorts
- Aspect Ratio: 9:16

Main Subject
- Person #2
- Black shirt

Keep
- Person #2

Blur
- Other faces
- License plates
```

사용자는 편집 실행 전에 Agent와 대화하며 계획을 수정할 수 있다.

이 기능은 다음 장점이 있다.

- Agent의 잘못된 대상 선택을 사전에 수정 가능
- 긴 영상 렌더링 실패 비용 감소
- AI 판단에 대한 사용자 신뢰 향상
- 대화형 Agent의 존재 이유를 시각적으로 보여줄 수 있음

---

# 4. 핵심 기능

프로젝트 범위를 제한하기 위해 기능은 총 3개로 고정한다.

---

## 4.1 Feature 1 — Person-Centered Auto Reframe

### 목적

16:9 등의 일반 영상을 특정 인물을 중심으로 9:16 쇼츠 비율로 자동 변환한다.

### 예시 명령

> "검은 옷 입은 사람 중심으로 쇼츠로 만들어줘."

### 처리 과정

```text
Natural Language Instruction
        ↓
Target Person Grounding
        ↓
Person Tracking
        ↓
Frame-by-frame Position
        ↓
Dynamic Crop Calculation
        ↓
9:16 Video Rendering
```

### 핵심 기능

- 자연어로 대상 인물 선택
- 대상 인물 bounding box 추적
- 대상 인물을 중심으로 crop position 계산
- 16:9 → 9:16 변환
- 영상 전체에서 부드러운 reframing 유지

### 구현 원칙

정밀 segmentation은 필수로 사용하지 않는다.

Auto Reframe은 bounding box 기반 tracking만으로도 충분히 구현 가능하도록 설계한다.

---

## 4.2 Feature 2 — Face / License Plate Privacy Blur

### 목적

영상 속 얼굴 및 자동차 번호판을 자동으로 탐지하고 영상 전체에서 추적하여 Blur 처리한다.

### 예시 명령

> "사람 얼굴이랑 번호판 전부 가려줘."

또는

> "검은 옷 남자는 남기고 다른 사람 얼굴은 가려줘."

### 처리 과정

```text
Video
 ↓
Face / License Plate Detection
 ↓
Object Tracking
 ↓
Target Filtering
 ↓
Bounding Box Blur
 ↓
Rendered Video
```

### 핵심 기능

- 얼굴 detection
- 차량 번호판 detection
- frame-to-frame tracking
- 특정 인물 제외 가능
- bounding box 기반 Blur

### 구현 원칙

프로젝트 난이도를 낮추기 위해 pixel-level segmentation보다 bounding box 기반 블러를 우선한다.

따라서 SAM 또는 diffusion 기반 편집은 핵심 의존성으로 두지 않는다.

---

## 4.3 Feature 3 — Conversational Edit Plan

### 목적

사용자의 자연어 명령을 구조화된 편집 계획으로 변환하고, 대화를 통해 수정한 뒤 실행한다.

### 예시

```text
User:
"검은 옷 남자 중심으로 쇼츠로 만들고 다른 얼굴이랑 번호판 가려줘."

Agent:
Edit Plan

1. Target Person
   - Black shirt man

2. Auto Reframe
   - 9:16

3. Privacy Blur
   - Other faces
   - License plates
```

사용자:

> "왼쪽 여자도 남겨줘."

Agent:

```text
Updated Edit Plan

Keep
- Black shirt man
- Left woman

Blur
- Other faces
- License plates
```

### 핵심 기능

- 자연어 명령 parsing
- Edit Plan 생성
- 대상 객체와 실제 영상 객체 연결
- 이전 대화 context 유지
- 사용자의 수정 명령 반영
- 사용자 승인 후 실제 Tool 실행

---

# 5. 시스템 아키텍처

```text
                        ┌───────────────┐
                        │     User      │
                        └───────┬───────┘
                                │
                        Natural Language
                                │
                                ▼
                  ┌────────────────────────┐
                  │ Conversational Agent   │
                  │                        │
                  │ - Intent Parsing       │
                  │ - Context Management   │
                  │ - Edit Planning        │
                  └───────────┬────────────┘
                              │
                              ▼
                         Edit Plan
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
      Visual Grounding                Editing Tools
               │                             │
       Target Identification                │
               │                    ┌────────┴─────────┐
               ▼                    │                  │
         Object Tracking        Auto Reframe      Privacy Blur
               │                    │                  │
               └────────────┬───────┴──────────────────┘
                            │
                            ▼
                      Video Renderer
                            │
                            ▼
                       Output Video
```

---

# 6. Agent 내부 구조

LLM은 영상 픽셀을 직접 편집하지 않는다.

LLM의 역할은 **Planner / Orchestrator**로 제한한다.

예를 들어 다음 명령이 들어온다고 가정한다.

> "검은 셔츠 남자 중심으로 쇼츠 만들고 나머지 얼굴과 번호판 가려줘."

Agent는 이를 다음과 같은 구조로 변환한다.

```json
{
  "target_subject": {
    "description": "black shirt man"
  },
  "reframe": {
    "enabled": true,
    "aspect_ratio": "9:16",
    "follow_target": true
  },
  "privacy": {
    "face_blur": true,
    "exclude_target_subject": true,
    "license_plate_blur": true
  }
}
```

그리고 필요한 tool을 호출한다.

```text
find_person("black shirt man")

track_person(person_id)

auto_reframe(person_id, "9:16")

detect_faces()

detect_license_plates()

blur_faces(exclude=[person_id])

blur_license_plates()

render_video()
```

---

# 7. Edit Plan Representation

Agent가 내부적으로 사용하는 편집 계획을 명시적으로 정의한다.

예시:

```json
{
  "video_id": "sample_001",
  "main_subjects": [
    {
      "object_id": "person_02",
      "description": "black shirt man"
    }
  ],
  "operations": [
    {
      "type": "AUTO_REFRAME",
      "target": "person_02",
      "aspect_ratio": "9:16"
    },
    {
      "type": "FACE_BLUR",
      "exclude": ["person_02"]
    },
    {
      "type": "LICENSE_PLATE_BLUR"
    }
  ]
}
```

Edit Plan은 다음 두 목적을 가진다.

1. Agent의 reasoning 결과를 실제 편집 도구가 실행 가능한 형태로 변환
2. 사용자에게 Agent의 편집 계획을 시각적으로 보여주기

---

# 8. 예상 기술 스택

## Frontend

- React / Next.js
- Video Preview UI
- Chat Interface
- Edit Plan Panel
- Bounding Box Overlay

## Backend

- Python
- FastAPI
- Background video processing pipeline

## Agent

- LLM API 또는 local instruction model
- Structured Output
- Tool Calling
- Conversation State Management

## Computer Vision

### Person / Object Grounding

후보:

- VLM
- Grounding DINO 계열
- Open-vocabulary detector

### Person Tracking

후보:

- ByteTrack
- BoT-SORT
- SAM2 Tracking (필요할 경우)

### Face Detection

후보:

- RetinaFace
- YOLO Face detector

### License Plate Detection

후보:

- pretrained YOLO detector
- 필요 시 별도 공개 데이터로 fine-tuning

## Video Processing

- OpenCV
- FFmpeg

---

# 9. 데이터 전략

본 프로젝트는 새로운 foundation model 학습을 목표로 하지 않는다.

따라서 전체 데이터셋을 직접 구축하거나 대규모 모델을 처음부터 학습하는 대신, pretrained model을 적극적으로 활용한다.

필요한 데이터는 다음 목적에 한해 사용한다.

### 번호판 Detector 보완

기존 모델의 한국 차량 번호판 성능이 부족할 경우 공개 데이터 또는 AI Hub 데이터를 이용하여 fine-tuning한다.

### 테스트 영상

다음 조건을 포함하는 자체 테스트 영상을 별도로 구축한다.

- 여러 명의 사람이 등장
- 서로 다른 색상의 옷
- 인물의 등장/퇴장
- 카메라 이동
- 차량 등장
- 번호판 노출

이를 통해 대화형 명령을 체계적으로 평가한다.

---

# 10. 서비스 UX

## 10.1 기본 화면

```text
┌───────────────────────────────────────────────┐
│                 EditAgent                     │
├────────────────────────┬──────────────────────┤
│                        │                      │
│                        │   Agent Chat         │
│                        │                      │
│     Video Preview      │  User Command        │
│                        │                      │
│                        │  Agent Response       │
│                        │                      │
├────────────────────────┼──────────────────────┤
│ Object Overlay         │ Edit Plan            │
│ - Person #1            │ - Reframe 9:16       │
│ - Person #2            │ - Blur Faces         │
│ - Plate #1             │ - Blur Plates        │
└────────────────────────┴──────────────────────┘
```

---

## 10.2 Workflow

### Step 1. Video Upload

사용자가 영상을 업로드한다.

### Step 2. Video Analysis

시스템이 영상에서 다음 객체를 분석한다.

- 사람
- 얼굴
- 번호판

### Step 3. Natural Language Command

사용자:

> "검은 옷 남자 중심으로 쇼츠로 만들고 다른 사람 얼굴이랑 번호판 가려줘."

### Step 4. Edit Plan Preview

Agent가 분석 결과를 보여준다.

### Step 5. Conversational Correction

사용자가 Agent 판단을 수정한다.

> "왼쪽 여자도 남겨줘."

### Step 6. Execution

사용자 승인 후 렌더링한다.

### Step 7. Before / After

원본과 결과 영상을 비교한다.

---

# 11. 대표 데모 시나리오

## Demo 1 — Auto Reframe

### Input

일반적인 16:9 인물 영상

### Command

> "검은 옷 입은 사람 중심으로 쇼츠로 만들어줘."

### Output

- 검은 옷 인물 추적
- 9:16 video conversion
- 인물이 중앙에 유지되는 dynamic crop

### 데모 포인트

16:9 → 9:16의 강한 시각적 Before / After

---

## Demo 2 — Privacy Blur

### Command

> "사람 얼굴이랑 차량 번호판 전부 가려줘."

### Output

- 얼굴 자동 추적 Blur
- 번호판 자동 추적 Blur

### 데모 포인트

영상 전체에서 Blur가 움직이는 객체를 따라가는 모습을 보여준다.

---

## Demo 3 — Conversational Complex Edit

### Command

> "검은 옷 입은 남자 중심으로 쇼츠로 만들고 다른 사람 얼굴이랑 차량 번호판은 가려줘."

Agent가 Edit Plan을 생성한다.

사용자:

> "왼쪽 여자도 남겨줘."

Agent가 Plan을 수정한다.

사용자:

> "이대로 편집해."

### 데모 포인트

본 프로젝트의 핵심 contribution을 가장 잘 보여주는 시나리오이다.

단순 영상처리가 아니라 **대화형 Agent를 이용한 편집 과정** 자체를 강조한다.

---

# 12. 평가 방법

프로젝트가 서비스 중심이므로 SOTA 모델 accuracy 경쟁보다 **명령 이해 및 최종 편집 성공 여부**를 중심으로 평가한다.

## 12.1 Instruction Understanding Accuracy

사용자의 명령을 올바른 Edit Plan으로 변환했는지 평가한다.

예:

```text
"사람 얼굴 전부 가려줘."

Expected:
FACE_BLUR = true
```

---

## 12.2 Target Grounding Accuracy

자연어로 지칭한 인물을 실제 영상 객체와 정확히 연결했는지 평가한다.

예:

```text
"검은 셔츠 남자"

Expected:
Person #2
```

---

## 12.3 Tracking Success Rate

선택된 인물을 영상 전체에서 안정적으로 추적하는지 평가한다.

---

## 12.4 Privacy Blur Success Rate

영상에서 노출된 얼굴/번호판 중 성공적으로 Blur 처리된 비율을 측정한다.

---

## 12.5 Task Completion Rate

자연어 명령이 최종 영상까지 올바르게 수행되었는지 End-to-End로 평가한다.

테스트 명령은 난이도별로 구성한다.

### Level 1 — Simple

> "얼굴 가려줘."

### Level 2 — Visual Reference

> "검은 옷 남자 중심으로 만들어줘."

### Level 3 — Exclusion

> "검은 옷 남자는 남기고 다른 얼굴 가려줘."

### Level 4 — Compositional

> "검은 옷 남자 중심으로 쇼츠로 만들고 다른 얼굴과 번호판은 가려줘."

---

# 13. 12주 개발 일정

## Week 1–2 — Core Video Pipeline

목표:

```text
Video
→ Detection
→ Tracking
→ Blur / Crop
→ MP4
```

구현:

- 영상 업로드
- face detection
- license plate detection
- person tracking
- bounding box blur
- 영상 렌더링

이 단계에서 Agent 없이 영상 처리 pipeline을 먼저 완성한다.

---

## Week 3–4 — Auto Reframe

목표:

```text
Person Tracking
→ Crop Center
→ 9:16 Reframe
```

구현:

- 대상 인물 선택
- frame별 위치 추적
- crop trajectory 계산
- smoothing
- 9:16 영상 생성

---

## Week 5–6 — Natural Language Agent

목표:

```text
Natural Language
→ Edit Plan
```

구현:

- LLM structured output
- tool schema
- Edit Plan JSON
- 간단한 multi-operation instruction 처리

---

## Week 7–8 — Conversational Editing

목표:

```text
User Command
→ Plan
→ User Correction
→ Updated Plan
```

구현:

- conversation state
- object reference 유지
- exclusion / keep 수정
- Edit Plan 변경

본 프로젝트의 가장 중요한 개발 구간이다.

---

## Week 9–10 — Web Service

구현:

- 영상 업로드 UI
- video preview
- Chat UI
- Edit Plan Panel
- object bounding box overlay
- rendering progress
- 결과 영상 preview

---

## Week 11 — Evaluation / Bug Fix

- 테스트 영상 구성
- instruction benchmark 작성
- grounding accuracy 측정
- tracking 안정성 테스트
- End-to-End task completion 측정
- 실패 사례 분석

---

## Week 12 — Presentation Polish

- UI 개선
- Before / After demo
- 발표용 테스트 영상
- 포스터
- 발표 자료
- 시연 시나리오 고정
- fallback demo 준비

이 시점에서는 새로운 기능을 추가하지 않는다.

---

# 14. 범위 제한

프로젝트 실패를 방지하기 위해 다음 기능은 구현 범위에서 제외한다.

## 제외 기능

- Object Removal
- Generative Video Inpainting
- BGM 자동 생성
- 생성형 B-roll
- 자동 Transition 생성
- 완전 자동 Highlight Detection
- Silence Removal
- Auto Caption Generation
- Adobe Premiere 수준 Timeline Editor
- 범용 자연어 영상 편집

이 기능들은 프로젝트의 핵심 contribution과 직접적으로 관련되지 않으며 개발 난이도를 불필요하게 높일 수 있다.

---

# 15. 예상 리스크 및 대응

## Risk 1 — 자연어 Grounding 실패

### 문제

"검은 옷 남자"와 같은 대상 지칭이 부정확할 수 있다.

### 대응

- Agent가 후보 인물을 화면에 표시
- 객체 ID 기반으로 사용자 확인 가능
- 사용자가 "왼쪽 사람"과 같이 수정 가능하도록 설계

---

## Risk 2 — Tracking ID Switch

### 문제

사람이 겹치거나 화면 밖으로 나갔다 다시 들어오면 tracking ID가 변경될 수 있다.

### 대응

- appearance embedding 활용
- re-identification 지원 tracker 검토
- 데모 영상은 지나치게 어려운 crowded scene을 피함

---

## Risk 3 — License Plate Detection 성능

### 문제

한국 번호판에 대해 pretrained detector 성능이 부족할 수 있다.

### 대응

- 필요 시 한국 번호판 공개 데이터 fine-tuning
- 프로젝트 전체 학습 범위를 번호판 detector 정도로 제한

---

## Risk 4 — 렌더링 시간

### 문제

긴 영상은 processing 시간이 오래 걸릴 수 있다.

### 대응

- 초기 서비스는 영상 길이를 제한
- Preview analysis와 final rendering 분리
- low-resolution preview → final resolution render 구조 검토

---

# 16. 성공 기준

프로젝트가 성공했다고 판단하기 위한 최소 조건은 다음과 같다.

### Essential

- [ ] 사용자가 영상 업로드 가능
- [ ] 자연어 명령 입력 가능
- [ ] Agent가 Edit Plan 생성 가능
- [ ] 대상 인물 자연어 선택 가능
- [ ] 특정 인물 중심 9:16 Auto Reframe 가능
- [ ] 얼굴 Blur 가능
- [ ] 번호판 Blur 가능
- [ ] 특정 인물 Blur 제외 가능
- [ ] 사용자가 Agent Plan을 대화로 수정 가능
- [ ] 최종 영상 렌더링 가능

### Presentation Quality

- [ ] 영상에서 대상 객체 표시
- [ ] Agent Edit Plan 시각화
- [ ] Before / After 비교
- [ ] 실제 대화형 수정 시연

---

# 17. 최종 발표 메시지

본 프로젝트를 다음과 같이 설명한다.

> 기존 영상 편집 도구에서는 사용자가 어떤 기능을 어디에 적용해야 하는지 직접 알아야 한다.
>
> EditAgent는 사용자가 원하는 편집 결과를 자연어로 설명하면, 영상의 시공간적 맥락과 객체를 이해하고 필요한 편집 작업을 계획하여 실행한다.
>
> 또한 Agent가 생성한 편집 계획을 사용자가 대화를 통해 수정할 수 있도록 하여, 영상 편집을 도구 중심의 작업에서 의도 중심의 상호작용으로 전환한다.

---

# 18. 최종 프로젝트 정의

## EditAgent
### Conversational AI Agent for Intent-Based Video Editing

**핵심 기능**

1. Person-Centered Auto Reframe
2. Face / License Plate Privacy Blur
3. Conversational Edit Plan

**핵심 Contribution**

> **복합적인 자연어 영상 편집 의도를 실제 영상 객체와 편집 operation으로 연결하고, 사용자와 대화를 통해 편집 계획을 수정·실행할 수 있는 Conversational Video Editing Agent**

**핵심 슬로건**

> **From Tool-Based Video Editing to Intent-Based Video Editing.**

