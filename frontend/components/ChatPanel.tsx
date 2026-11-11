"use client";
import { useEffect, useRef } from "react";
import type { Question } from "@/types/api";
import type { Message } from "@/types/editor";

type Props = {
  messages: Message[];
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  disabled: boolean;
  busy: string | null;
  /** Id of the message whose question is still open; only it shows answer buttons. */
  openQuestionMessage?: number;
  onAnswer: (question: Question, option: string, label: string) => void;
  /** Id of the message that shows the current spec; it gets the approve button. */
  specMessage?: number;
  canApprove: boolean;
  onApprove: () => void;
};

export default function ChatPanel({ messages, input, setInput, onSend, disabled, busy, openQuestionMessage, onAnswer, specMessage, canApprove, onApprove }: Props) {
  const end = useRef<HTMLDivElement | null>(null);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [messages, busy]);

  return (
    <aside className="chat">
      <div className="chat-header">
        <h2>AI 편집 어시스턴트</h2>
        <span className="demo">계획 → 확인 → 실행</span>
      </div>
      <div className="messages" aria-live="polite">
        {messages.map(message => (
          <div key={message.id} className={`message-row ${message.sender}`}>
            <div className={`message ${message.tone === "error" ? "message-error" : ""}`}>
              {message.text}
              {message.question && message.id === openQuestionMessage && (
                <div className="answer-options">
                  {message.question.options.map(option => (
                    <button key={option.id} type="button" className="answer-option" disabled={!!busy}
                      onClick={() => onAnswer(message.question!, option.id, option.label)}>
                      {option.label}
                    </button>
                  ))}
                  {(message.question.kind === "choose" || message.question.kind === "confirm") && (
                    <span className="answer-hint">화면의 박스를 클릭해서 고를 수도 있어요.</span>
                  )}
                </div>
              )}
              {message.id === specMessage && (
                <div className="answer-options">
                  <button type="button" className="button" disabled={!canApprove} onClick={onApprove}>승인하고 렌더링</button>
                  <span className="answer-hint">고칠 게 있으면 그냥 말씀하시거나 화면을 클릭하세요.</span>
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && <div className="message-row assistant"><div className="message message-busy">{busy}</div></div>}
        <div ref={end} />
      </div>
      <form className="chat-form" onSubmit={e => { e.preventDefault(); onSend(); }}>
        <textarea value={input} onChange={e => setInput(e.target.value)} disabled={disabled} rows={2}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); onSend(); } }}
          placeholder={disabled && !busy ? "먼저 영상을 올려 주세요" : "편집 계획이나 답을 입력하세요 (Shift+Enter 줄바꿈)"} aria-label="편집 요청" />
        <button className="button" type="submit" disabled={disabled || !input.trim()}>전송</button>
      </form>
    </aside>
  );
}
