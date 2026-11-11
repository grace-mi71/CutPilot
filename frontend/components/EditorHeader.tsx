"use client";

import { useRef, useState } from "react";

type Props = {
  provider: string | null;
  canExport: boolean;
  onExport: () => void;
  textOnly: boolean;
};

export default function EditorHeader({ provider, canExport, onExport, textOnly }: Props) {
  const [title, setTitle] = useState("영상 편집 1");
  const [draft, setDraft] = useState(title);
  const [editing, setEditing] = useState(false);
  const cancelEdit = useRef(false);

  function saveTitle() {
    if (cancelEdit.current) {
      setDraft(title);
      cancelEdit.current = false;
    } else {
      setTitle(draft.trim() || title);
    }
    setEditing(false);
  }

  return (
    <header className="header">
      <strong className="logo">CutPilot</strong>
      <div className="header-center">
        {editing ? (
          <span className="project-title-editor">
            <span className="project-title-sizer" aria-hidden="true">{draft || "영상 제목"}</span>
            <input
              className="project-title project-title-input"
              aria-label="영상 제목"
              value={draft}
              maxLength={80}
              autoFocus
              onFocus={event => event.currentTarget.select()}
              onChange={(event) => setDraft(event.target.value)}
              onBlur={saveTitle}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.nativeEvent.isComposing) event.currentTarget.blur();
                if (event.key === "Escape") {
                  cancelEdit.current = true;
                  event.currentTarget.blur();
                }
              }}
            />
          </span>
        ) : (
          <button
            type="button"
            className="project-title project-title-button"
            title="제목 수정"
            aria-label={`영상 제목 수정: ${title}`}
            onClick={() => { cancelEdit.current = false; setDraft(title); setEditing(true); }}
          >
            <span>{title}</span><span className="title-edit-icon" aria-hidden="true">✎</span>
          </button>
        )}
        {provider === "offline-keywords" && <span className="badge" title="AI 없이 키워드 규칙으로 동작합니다">오프라인 모드</span>}
        {textOnly && <span className="badge" title="미리보기의 박스를 숨긴 실험 조건">텍스트 전용</span>}
      </div>
      <button className="button" disabled={!canExport} onClick={onExport}
        title={canExport ? "원본 해상도로 렌더링해 새 탭에서 엽니다" : "질문에 모두 답하면 내보낼 수 있어요"}>
        내보내기
      </button>
    </header>
  );
}
