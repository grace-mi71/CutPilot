type Props = {
  videoUrl: string | null;
  onSelect: (file: File) => void;
  selectPrompt: boolean;
  selecting: boolean;
  onCancelSelection: () => void;
  view: "original" | "complete";
  onViewChange: (view: "original" | "complete") => void;
};

export default function VideoPreview({ videoUrl, onSelect, selectPrompt, selecting, onCancelSelection, view, onViewChange }: Props) {
  return (
    <>
      <div className="tabs">
        <button type="button" className={`tab ${view === "original" ? "active" : ""}`} aria-pressed={view === "original"} onClick={() => onViewChange("original")}>▣ &nbsp; 원본</button>
        <button type="button" className={`tab ${view === "complete" ? "active" : ""}`} aria-pressed={view === "complete"} onClick={() => onViewChange("complete")}>◎ &nbsp; 완료</button>
        {selecting && <button className="cancel-selection" type="button" onClick={onCancelSelection}>선택 취소</button>}
      </div>
      <div className={`video-wrapper ${selecting ? "video-highlight" : ""}`}>
        {view === "complete" ? (
          <div className="empty-video">
            <div className="video-icon">▶</div>
            <strong>아직 완료된 영상이 없어요</strong>
            <p>편집과 렌더링이 완료되면 이곳에서 결과를 확인할 수 있어요.</p>
          </div>
        ) : videoUrl ? (
          <video className="video" controls src={videoUrl} />
        ) : (
          <div className="empty-video">
            <div className="video-icon">▶</div>
            <strong>편집할 영상을 선택해 주세요</strong>
            <p>선택한 영상은 이 브라우저에서만 미리 볼 수 있어요.</p>
            <label className="button upload">영상 선택<input type="file" accept="video/*" onChange={e => { const file = e.target.files?.[0]; if (file) onSelect(file); }} /></label>
          </div>
        )}
        <div className={`selection-message ${selectPrompt ? "is-visible" : ""}`} role="status" aria-live="polite" aria-hidden={!selectPrompt}>
          <span>편집하고 싶은 객체를 선택해주세요.</span>
        </div>
      </div>
      {view === "original" && videoUrl && <label className="replace">다른 영상 선택<input type="file" accept="video/*" onChange={e => { const file = e.target.files?.[0]; if (file) onSelect(file); }} /></label>}
    </>
  );
}
