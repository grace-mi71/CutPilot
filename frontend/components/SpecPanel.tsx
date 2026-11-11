import type { Spec } from "@/types/api";

type Props = {
  spec: Spec | null;
  canApprove: boolean;
  onApprove: () => void;
};

export default function SpecPanel({ spec, canApprove, onApprove }: Props) {
  return (
    <aside className="spec-panel" aria-labelledby="spec-panel-title">
      <div className="spec-panel-heading">
        <h2 id="spec-panel-title">작업 명세서</h2>
        <p>편집을 시작하기 전에 내용을 확인하세요.</p>
      </div>

      <div className="spec-panel-body">
        {spec ? (
          <>
            {spec.steps.length ? (
              <ol className="spec-steps">
                {spec.steps.map((step, position) => (
                  <li className="spec-step" key={step.index}>
                    <div className="spec-step-heading">
                      <span className="spec-step-number">{String(position + 1).padStart(2, "0")}</span>
                      <h3>{step.label}</h3>
                    </div>
                    <dl className="spec-step-details">
                      <div><dt>대상</dt><dd>{step.targets.join(", ") || "대상 없음"}</dd></div>
                      <div><dt>구간</dt><dd>{step.time}</dd></div>
                      {step.excluded.length > 0 && (
                        <div><dt>제외</dt><dd>{step.excluded.join(", ")}</dd></div>
                      )}
                    </dl>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="spec-empty">모든 작업이 적용 안 함 상태예요. 렌더링하면 원본 영상이 그대로 나옵니다.</p>
            )}
            {spec.decisions.length > 0 && (
              <section className="spec-decisions" aria-labelledby="spec-decisions-title">
                <h3 id="spec-decisions-title">확인한 내용</h3>
                <ul>{spec.decisions.map((decision, index) => <li key={index}>{decision}</li>)}</ul>
              </section>
            )}
          </>
        ) : (
          <p className="spec-empty">AI와 편집 계획을 정하고 질문에 답하면 명세서가 여기에 표시돼요.</p>
        )}
      </div>

      {spec && (
        <div className="spec-panel-footer">
          <button className="button" type="button" disabled={!canApprove} onClick={onApprove}>
            승인하고 렌더링
          </button>
        </div>
      )}
    </aside>
  );
}
