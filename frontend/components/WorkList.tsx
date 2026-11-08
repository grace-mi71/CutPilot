import type { PlanWarning, Spec } from "@/types/api";
import type { WorkItem } from "@/types/editor";

type Props = {
  works: WorkItem[];
  onAdd: () => void;
  onRemove: (index: number) => void;
  readOnly: boolean;
  selected: number;
  onSelectRow: (index: number) => void;
  spec: Spec | null;
  canApprove: boolean;
  onApprove: () => void;
  warnings: PlanWarning[];
};

const STATUS: Record<WorkItem["status"], { label: string; className: string }> = {
  ok: { label: "확정", className: "status-ok" },
  confirm: { label: "확인 필요", className: "status-confirm" },
  choose: { label: "선택 필요", className: "status-choose" },
};

export default function WorkList({ works, onAdd, onRemove, readOnly, selected, onSelectRow, spec, canApprove, onApprove, warnings }: Props) {
  return (
    <section className="work-list">
      <div className="work-heading">
        <div>
          <h2>작업 리스트</h2>
          <p className="hint">AI가 이해한 계획입니다. 행을 고른 뒤 화면의 사람을 클릭하면 대상을 넣거나 뺄 수 있어요.</p>
        </div>
        {!readOnly && <button className="button" onClick={onAdd} title="화면에서 사람을 골라 채팅에 넣습니다">＋ &nbsp; 추가하기</button>}
      </div>
      <div className="work-scroll" tabIndex={0} role="region" aria-label="작업 목록">
        <table>
          <thead><tr><th>객체</th><th>인식 구간</th><th>수행할 작업</th><th>상태</th>{!readOnly && <th aria-label="작업 제거" />}</tr></thead>
          <tbody>
            {works.map(work => (
              <tr key={work.index} className={work.index === selected ? "row-selected" : ""} onClick={() => onSelectRow(work.index)}>
                <td><span className="avatar">{work.index + 1}</span><strong>{work.object}</strong></td>
                <td className="muted">{work.time}</td>
                <td>{work.action}</td>
                <td>
                  <span className={`status ${STATUS[work.status].className}`}>{STATUS[work.status].label}</span>
                  {work.warnings > 0 && <span className="status status-warn" title={warnings.filter(w => w.operation_index === work.index).map(w => w.message).join("\n")}>⚠ {work.warnings}</span>}
                </td>
                {!readOnly && <td className="remove-cell">
                  <button className="remove-work" type="button" aria-label={`${work.index + 1}번 작업 제거`} title="작업 제거"
                    onClick={e => { e.stopPropagation(); onRemove(work.index); }}>−</button>
                </td>}
              </tr>
            ))}
            {works.length === 0 && <tr><td colSpan={readOnly ? 4 : 5} className="muted">아직 계획이 없습니다. 채팅으로 편집 계획을 말씀해 주세요.</td></tr>}
          </tbody>
        </table>
        {spec && (
          <div className="spec-card">
            <h3>최종 명세서</h3>
            <ol>
              {spec.steps.map(step => (
                <li key={step.index}>
                  <strong>{step.label}</strong> — {step.targets.join(", ") || "대상 없음"}
                  {step.excluded.length > 0 && <> (제외: {step.excluded.join(", ")})</>} — {step.time}
                </li>
              ))}
            </ol>
            {spec.decisions.length > 0 && (
              <>
                <p className="spec-sub">확인한 내용</p>
                <ul>{spec.decisions.map((d, i) => <li key={i}>{d}</li>)}</ul>
              </>
            )}
            <button className="button" type="button" disabled={!canApprove} onClick={onApprove}>승인하고 렌더링</button>
          </div>
        )}
      </div>
    </section>
  );
}
