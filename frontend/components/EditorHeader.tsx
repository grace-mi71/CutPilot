"use client";

import { useState } from "react";

export default function EditorHeader() {
  const [title, setTitle] = useState("영상 편집 1");
  const [draft, setDraft] = useState(title);
  const [editing, setEditing] = useState(false);

  function saveTitle() {
    setTitle(draft.trim() || title);
    setEditing(false);
  }

  return (
    <header className="header">
      <strong className="logo">CutPilot</strong>
      {editing ? (
        <input
          className="project-title project-title-input"
          aria-label="영상 제목"
          value={draft}
          maxLength={80}
          autoFocus
          onChange={(event) => setDraft(event.target.value)}
          onBlur={saveTitle}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.nativeEvent.isComposing) event.currentTarget.blur();
            if (event.key === "Escape") {
              setDraft(title);
              setEditing(false);
            }
          }}
        />
      ) : (
        <button
          type="button"
          className="project-title project-title-button"
          title="제목 수정"
          aria-label={`영상 제목 수정: ${title}`}
          onClick={() => { setDraft(title); setEditing(true); }}
        >
          <span>{title}</span><span className="title-edit-icon" aria-hidden="true">✎</span>
        </button>
      )}
      <button className="button" disabled title="렌더링 연결 후 사용할 수 있어요">↥ &nbsp; 내보내기</button>
    </header>
  );
}
