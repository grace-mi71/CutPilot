import type { WorkItem } from "@/types/editor";

type Props = {
  works: WorkItem[];
  onAdd: () => void;
  onRemove: (id: number) => void;
  readOnly: boolean;
};

export default function WorkList({ works, onAdd, onRemove, readOnly }: Props) {
  return (
    <section className="work-list">
      <div className="work-heading">
        <div><h2>작업 리스트</h2><p className="hint">채팅에서 확인한 작업을 표시합니다.</p></div>
        {!readOnly && <button className="button" onClick={onAdd}>＋ &nbsp; 추가하기</button>}
      </div>
      <div className="work-scroll" tabIndex={0} role="region" aria-label="작업 목록">
        <table>
          <thead><tr><th>객체</th><th>인식 구간</th><th>수행할 작업</th>{!readOnly && <th aria-label="작업 제거" />}</tr></thead>
          <tbody>
            {works.map(work => (
              <tr key={work.id}>
                <td><span className="avatar">●</span><strong>{work.object}</strong></td>
                <td className="muted">{work.time}</td>
                <td>{work.action}</td>
                {!readOnly && <td className="remove-cell">
                  <button className="remove-work" type="button" aria-label={`${work.object} ${work.action} 작업 제거`} title="작업 제거" onClick={() => onRemove(work.id)}>−</button>
                </td>}
              </tr>
            ))}
            {works.length === 0 && <tr><td colSpan={readOnly ? 3 : 4} className="muted">추가된 작업이 없습니다.</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}
