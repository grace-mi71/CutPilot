"use client";
import { useEffect, useState, type RefObject } from "react";
import type { Box } from "@/types/api";

/** A person box drawn over the paused preview, coloured by what the plan does to them. */
export type BoxView = {
  trackId: string;
  box: Box;
  role: "target" | "candidate" | "asked" | "excluded" | "reframe_subject" | "none";
};

type Props = {
  videoUrl: string | null;
  renderUrl: string | null;
  onSelect: (file: File) => void;
  selectPrompt: boolean;
  selecting: boolean;
  onCancelSelection: () => void;
  view: "original" | "complete";
  onViewChange: (view: "original" | "complete") => void;
  videoRef: RefObject<HTMLVideoElement | null>;
  boxes: BoxView[];
  onBoxClick: (trackId: string) => void;
  onPaused: () => void;
  onPlaying: () => void;
  busy: string | null;
};

const ROLE_LABEL: Record<BoxView["role"], string> = {
  target: "편집 대상",
  candidate: "후보",
  asked: "이 사람?",
  excluded: "제외",
  reframe_subject: "세로 변환 기준",
  none: "",
};

/** Where the picture actually sits inside the element (object-fit: contain letterboxes it). */
function contentRect(el: HTMLVideoElement) {
  const w = el.clientWidth, h = el.clientHeight;
  const vw = el.videoWidth || w, vh = el.videoHeight || h;
  const scale = Math.min(w / vw, h / vh);
  const cw = vw * scale, ch = vh * scale;
  return { x: (w - cw) / 2, y: (h - ch) / 2, w: cw, h: ch };
}

export default function VideoPreview({
  videoUrl, renderUrl, onSelect, selectPrompt, selecting, onCancelSelection, view, onViewChange,
  videoRef, boxes, onBoxClick, onPaused, onPlaying, busy,
}: Props) {
  const [rect, setRect] = useState({ x: 0, y: 0, w: 0, h: 0 });

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    const update = () => setRect(contentRect(el));
    const observer = new ResizeObserver(update);
    observer.observe(el);
    el.addEventListener("loadedmetadata", update);
    return () => {
      observer.disconnect();
      el.removeEventListener("loadedmetadata", update);
    };
  }, [videoRef, videoUrl, view]);

  const picker = (label: string, className: string) => (
    <label className={className}>
      {label}
      <input type="file" accept="video/*" disabled={!!busy}
        onChange={e => { const file = e.target.files?.[0]; if (file) onSelect(file); e.target.value = ""; }} />
    </label>
  );

  return (
    <>
      <div className="tabs">
        <button type="button" className={`tab ${view === "original" ? "active" : ""}`} aria-pressed={view === "original"} onClick={() => onViewChange("original")}>원본</button>
        <button type="button" className={`tab ${view === "complete" ? "active" : ""}`} aria-pressed={view === "complete"} onClick={() => onViewChange("complete")}>완료</button>
        {selecting && <button className="cancel-selection" type="button" onClick={onCancelSelection}>선택 취소</button>}
      </div>
      <div className={`video-wrapper ${selecting ? "video-highlight" : ""}`}>
        {view === "complete" ? (
          renderUrl ? (
            <video key={renderUrl} className="video" controls src={renderUrl} />
          ) : (
            <div className="empty-video">
              <div className="video-icon">▶</div>
              <strong>아직 완료된 영상이 없어요</strong>
              <p>질문에 모두 답하고 명세서를 승인하면 이곳에서 결과를 확인할 수 있어요.</p>
            </div>
          )
        ) : videoUrl ? (
          <>
            <video
              ref={videoRef}
              className="video"
              controls
              src={videoUrl}
              onPause={onPaused}
              onSeeked={onPaused}
              onLoadedData={onPaused}
              onPlay={onPlaying}
            />
            <div className="box-layer" style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}>
              {boxes.map(b => (
                <button
                  key={b.trackId}
                  type="button"
                  className={`track-box role-${b.role}`}
                  style={{ left: `${b.box.x * 100}%`, top: `${b.box.y * 100}%`, width: `${b.box.w * 100}%`, height: `${b.box.h * 100}%` }}
                  title={`${b.trackId}${ROLE_LABEL[b.role] ? ` · ${ROLE_LABEL[b.role]}` : ""}`}
                  aria-label={`${b.trackId} 선택`}
                  onClick={() => onBoxClick(b.trackId)}
                >
                  <span>{b.trackId.replace("person_", "#")}{ROLE_LABEL[b.role] ? ` ${ROLE_LABEL[b.role]}` : ""}</span>
                </button>
              ))}
            </div>
          </>
        ) : (
          <div className="empty-video">
            <div className="video-icon">▶</div>
            <strong>편집할 영상을 선택해 주세요</strong>
            <p>영상은 분석을 위해 CutPilot 서버로 업로드됩니다. (최대 180초)</p>
            {picker("영상 선택", "button upload")}
          </div>
        )}
        {busy && <div className="busy-overlay" role="status"><span>{busy}</span></div>}
        <div className={`selection-message ${selectPrompt ? "is-visible" : ""}`} role="status" aria-live="polite" aria-hidden={!selectPrompt}>
          <span>일시정지한 화면에서 편집할 사람을 클릭하세요.</span>
        </div>
      </div>
      {view === "original" && videoUrl && picker("다른 영상 선택", "replace")}
    </>
  );
}
