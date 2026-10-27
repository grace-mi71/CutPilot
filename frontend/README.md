# CutPilot 프론트엔드

Next.js와 React로 만든 영상 편집 화면의 초기 프로토타입입니다.

## 실행

`frontend` 폴더에서 다음 명령을 실행한 뒤 브라우저에서 http://localhost:3000 을 엽니다.

```bash
npm install
npm run dev
```

## 현재 구현

- 원본 영상 파일을 브라우저에서 선택하고 재생
- 원본/완료 탭과 읽기 전용 완료 화면
- 예시 작업 목록, 삭제, 객체 선택 안내 및 선택 취소
- 예시 응답을 보여 주는 채팅 화면
- 작업 목록과 채팅의 독립적인 스크롤

현재 작업 목록과 채팅은 예시 데이터입니다. 객체 탐지·선택, 서버 업로드, AI 대화, 렌더링 및 결과 영상은 아직 백엔드와 연결되지 않았습니다. 선택한 원본 영상은 브라우저에서만 재생됩니다.

백엔드 연동 시 `backend/cutpilot/schemas/`의 VideoAsset, Track, EditPlan을 참고해 API 계약을 정합니다. 기존 프론트엔드 계획은 `docs/decisions/0001-initial-stack.md`를 참고하세요.
