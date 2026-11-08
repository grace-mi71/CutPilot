type Props = {
  provider: string | null;
  canExport: boolean;
  onExport: () => void;
  textOnly: boolean;
};

export default function EditorHeader({ provider, canExport, onExport, textOnly }: Props) {
  return (
    <header className="header">
      <strong className="logo">CutPilot</strong>
      <span className="project-title">
        대화형 영상 편집
        {provider === "offline-keywords" && <span className="badge" title="AI 없이 키워드 규칙으로 동작합니다">오프라인 모드</span>}
        {textOnly && <span className="badge" title="미리보기의 박스를 숨긴 실험 조건">텍스트 전용</span>}
      </span>
      <button className="button" disabled={!canExport} onClick={onExport}
        title={canExport ? "원본 해상도로 렌더링해 새 탭에서 엽니다" : "질문에 모두 답하면 내보낼 수 있어요"}>
        ↥ &nbsp; 내보내기
      </button>
    </header>
  );
}
