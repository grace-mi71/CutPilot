import type { PlanWarning } from "@/types/api";
import type { WorkItem } from "@/types/editor";

type Props = {
  works: WorkItem[];
  onAdd: () => void;
  onRemove: (index: number) => void;
  onToggleEnabled: (index: number, enabled: boolean) => void;
  view: "original" | "complete";
  hasRender: boolean;
  readOnly: boolean;
  canAdd: boolean;
  busy: boolean;
  selected: number;
  onSelectRow: (index: number) => void;
  warnings: PlanWarning[];
};

const STATUS: Record<WorkItem["status"], { label: string; className: string }> = {
  ok: { label: "확정", className: "status-ok" },
  confirm: { label: "확인 필요", className: "status-confirm" },
  choose: { label: "선택 필요", className: "status-choose" },
};

export default function WorkList({ works, onAdd, onRemove, onToggleEnabled, view, hasRender, readOnly, canAdd, busy, selected, onSelectRow, warnings }: Props) {
  const showActions = !readOnly && (view === "original" || hasRender);
  return (
    <section className="work-list">
      <div className="work-heading">
        <div>
          <h2>작업 리스트</h2>
          <p className="hint">AI가 이해한 편집 계획을 여기서 확인할 수 있어요.</p>
        </div>
        {canAdd && <button className="button button-secondary" onClick={onAdd} disabled={busy} title="화면에서 사람을 골라 채팅에 넣습니다">대상 추가</button>}
      </div>
      <div className="work-scroll" tabIndex={0} role="region" aria-label="작업 목록">
        <table>
          <thead><tr><th>객체</th><th>인식 구간</th><th>수행할 작업</th><th>상태</th>{showActions && <th aria-label={view === "complete" ? "작업 적용" : "작업 삭제"} />}</tr></thead>
          <tbody>
            {works.map(work => (
              <tr key={work.index} className={`${work.index === selected ? "row-selected" : ""} ${!work.enabled ? "row-disabled" : ""}`} onClick={() => onSelectRow(work.index)}>
                <td><span className="avatar">{work.index + 1}</span><strong>{work.object}</strong></td>
                <td className="muted">{work.time}</td>
                <td>{work.action}</td>
                <td>
                  {work.enabled ? (
                    <span className={`status ${STATUS[work.status].className}`}>{STATUS[work.status].label}</span>
                  ) : (
                    <span className="status status-off">적용 안 함</span>
                  )}
                  {work.enabled && <span className={`status ${work.rendered ? "status-rendered" : "status-pending"}`}>
                    {work.rendered ? "렌더링됨" : "렌더링 전"}
                  </span>}
                  {work.warnings > 0 && <span className="status status-warn" title={warnings.filter(w => w.operation_index === work.index).map(w => w.message).join("\n")}>⚠ {work.warnings}</span>}
                </td>
                {showActions && <td className="remove-cell">
                  <div className="work-actions">
                    {view === "complete" ? <button className={`visibility-work ${work.enabled ? "" : "is-off"}`} type="button"
                      aria-label={`${work.index + 1}번 작업 ${work.enabled ? "적용 중" : "적용 안 함"}`}
                      aria-pressed={work.enabled} title={work.enabled ? "적용 안 함" : "적용하기"}
                      disabled={busy}
                      onClick={e => { e.stopPropagation(); onToggleEnabled(work.index, !work.enabled); }}>
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M2.5 12s3.5-5 9.5-5 9.5 5 9.5 5-3.5 5-9.5 5-9.5-5-9.5-5Z" />
                        <circle cx="12" cy="12" r="2.5" />
                        {!work.enabled && <path d="M3 3 21 21" />}
                      </svg>
                    </button> : <button className="remove-work" type="button" aria-label={`${work.index + 1}번 작업 제거`} title="작업 제거" disabled={busy}
                      onClick={e => { e.stopPropagation(); onRemove(work.index); }}>−</button>}
                  </div>
                </td>}
              </tr>
            ))}
            {works.length === 0 && <tr><td colSpan={showActions ? 5 : 4} className="muted">아직 계획이 없습니다. 채팅으로 편집 계획을 말씀해 주세요.</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}
