"use client";

import { useEffect, useRef } from "react";

type Props = {
  busy: string | null;
  error: string | null;
  canRetry: boolean;
  onSelect: (file: File) => void;
  onRetry: () => void;
  onClose: () => void;
};

export default function Onboarding({ busy, error, canRetry, onSelect, onRetry, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    element.showModal();
    return () => { if (element.open) element.close(); };
  }, []);

  return (
    <dialog
      ref={dialog}
      className="onboarding-dialog"
      aria-labelledby="onboarding-title"
      onCancel={event => { event.preventDefault(); onClose(); }}
    >
      <header className="onboarding-header">
        <strong className="onboarding-logo">CutPilot</strong>
        <span className="onboarding-header-label">How to use</span>
        <button type="button" className="onboarding-close" aria-label="안내 닫기" onClick={onClose}>×</button>
      </header>

      <div className="onboarding-content">
        <section className="onboarding-intro" aria-labelledby="onboarding-title">
          <h1 id="onboarding-title">영상에서 바꾸고 싶은 점을 말해 주세요</h1>
          <p>원하는 결과를 말하면 AI가 계획을 제안해요. 확인하고 승인하면 편집이 시작돼요.</p>
        </section>

        <section className="onboarding-example" aria-labelledby="example-title">
          <h2 id="example-title">사용 예시</h2>
          <div className="onboarding-example-cards">
            <article className="onboarding-example-card request-card">
              <h3>내 요청</h3>
              <p>“공개할 영상인데, 지나가는 사람들이 누군지 알아볼 수 없게 해줘.”</p>
            </article>
            <article className="onboarding-example-card proposal-card">
              <h3>AI의 편집 제안</h3>
              <p>“영상 전체에서 감지된 사람들에게 모자이크를 적용해요.”</p>
            </article>
          </div>
        </section>

        <section className="onboarding-flow" aria-labelledby="flow-title">
          <h2 id="flow-title">계획부터 완료까지</h2>
          <ol>
            <li><strong>계획 확인</strong><span>왼쪽 명세서와 렌더링 상태를 확인해요. 원본 탭의 −로 작업을 뺄 수 있어요.</span></li>
            <li><strong>완료 후 조정</strong><span>완료 탭의 눈으로 작업을 숨기거나 복원해요. 목록에 남긴 채 다시 렌더링하고 내보내요.</span></li>
          </ol>
        </section>

        <section className="onboarding-capabilities" aria-labelledby="capabilities-title">
          <h2 id="capabilities-title">지금 가능한 편집</h2>
          <ul>
            <li><strong>모자이크·블러</strong><span>사람 전체 또는 얼굴 영역</span></li>
            <li><strong>대상·구간 지정</strong><span>전체 또는 특정 사람과 시간대</span></li>
            <li><strong>세로 영상 변환</strong><span>한 사람을 따라가는 9:16 화면 · 모자이크와 함께 적용</span></li>
          </ul>
          <p>얼굴 영역은 위치를 추정해요. 번호판 편집은 아직 지원하지 않아요.</p>
        </section>
      </div>

        <footer className="onboarding-start">
          <input
            ref={fileInput}
            className="onboarding-file-input"
            type="file"
            accept="video/*"
            aria-label="편집할 영상 선택"
            tabIndex={-1}
            disabled={!!busy}
            onChange={event => {
              const file = event.target.files?.[0];
              if (file) onSelect(file);
              event.target.value = "";
            }}
          />
          <button
            type="button"
            className="onboarding-primary-button"
            autoFocus
            disabled={!!busy}
            onClick={() => fileInput.current?.click()}
          >
            영상 올리고 시작하기
          </button>
          {busy ? (
            <p className="onboarding-progress" role="status" aria-live="polite">
              <span className="onboarding-spinner" aria-hidden="true" />{busy}
            </p>
          ) : error ? (
            <div className="onboarding-error" role="alert">
              <p>업로드를 완료하지 못했어요. {error}</p>
              <p>서버 연결과 영상 파일을 확인한 뒤 다시 시도해 주세요.</p>
              {canRetry && <button type="button" onClick={onRetry}>같은 영상 다시 시도</button>}
            </div>
          ) : (
            <p className="onboarding-file-note">분석 후 편집 화면으로 이동해요 · 최대 180초 영상</p>
          )}
        </footer>
    </dialog>
  );
}
