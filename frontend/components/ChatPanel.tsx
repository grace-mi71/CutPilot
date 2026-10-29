import { useEffect, useRef } from "react";
import type { Message } from "@/types/editor";

type Props = {
  messages: Message[];
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  sending: boolean;
  hasPendingPlan: boolean;
  onConfirmPlan: () => void;
};

export default function ChatPanel({ messages, input, setInput, onSend, sending, hasPendingPlan, onConfirmPlan }: Props) {
  const messagesRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = messagesRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [messages, sending, hasPendingPlan]);

  return (
    <aside className="chat">
      <div className="chat-header"><h2>AI 편집 어시스턴트</h2><span className="demo">● &nbsp; 오프라인 모드</span></div>
      <p className="offline-notice">현재 가상 객체를 사용하며, 실제 영상에는 적용되지 않습니다.</p>
      <div className="messages" ref={messagesRef} aria-live="polite">
        {messages.map(message => <div key={message.id} className={`message-row ${message.sender}`}><div className="message">{message.text}</div></div>)}
        {sending && <div className="message-row assistant"><div className="message">응답을 준비하고 있어요…</div></div>}
        {hasPendingPlan && <button type="button" className="button confirm-plan" onClick={onConfirmPlan}>확인 후 작업 리스트에 반영</button>}
      </div>
      <form className="chat-form" onSubmit={e => { e.preventDefault(); onSend(); }}>
        <input value={input} onChange={e => setInput(e.target.value)} placeholder="예: person_02 블러" aria-label="편집 요청" />
        <button className="button" type="submit" disabled={sending || !input.trim()}>전송 ↗</button>
      </form>
    </aside>
  );
}
