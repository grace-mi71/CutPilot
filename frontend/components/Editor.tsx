"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import EditorHeader from "./EditorHeader";
import VideoPreview, { type BoxView } from "./VideoPreview";
import WorkList from "./WorkList";
import ChatPanel from "./ChatPanel";
import { initialMessages } from "@/data/mockData";
import { api, ApiError, fileUrl } from "@/lib/api";
import type { Confirmation, Question, Render, Reply, Spec, TrackBox, Video } from "@/types/api";
import type { Message, WorkItem } from "@/types/editor";

let nextId = 100;
const newId = () => ++nextId;

function rowsOf(confirmation: Confirmation | null): WorkItem[] {
  if (!confirmation) return [];
  return confirmation.plan.operations.map((_, index) => {
    const items = confirmation.items.filter(item => item.operation_index === index);
    const field = (name: string) => items.find(item => item.field === name);
    return {
      index,
      object: field("selector")?.label ?? "",
      time: field("time_range")?.label ?? "",
      action: field("op")?.label ?? "",
      status: field("selector")?.status ?? "ok",
      warnings: confirmation.warnings.filter(w => w.operation_index === index).length,
    };
  });
}

export default function Editor({ textOnly = false }: { textOnly?: boolean }) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [view, setView] = useState<"original" | "complete">("original");
  const [selectPrompt, setSelectPrompt] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const promptTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [video, setVideo] = useState<Video | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [provider, setProvider] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [question, setQuestion] = useState<Question | null>(null);
  const [spec, setSpec] = useState<Spec | null>(null);
  const [canApprove, setCanApprove] = useState(false);
  const [render, setRender] = useState<Render | null>(null);
  const [selectedRow, setSelectedRow] = useState(0);

  const videoEl = useRef<HTMLVideoElement | null>(null);
  const [boxes, setBoxes] = useState<TrackBox[]>([]);

  useEffect(() => () => { if (promptTimer.current) clearTimeout(promptTimer.current); }, []);

  const say = useCallback((sender: Message["sender"], text: string, extra: Partial<Message> = {}) => {
    setMessages(old => [...old, { id: newId(), sender, text, ...extra }]);
  }, []);

  const refreshBoxes = useCallback(async () => {
    const el = videoEl.current;
    if (!video || !el || textOnly) return;
    try {
      setBoxes((await api.boxes(video.id, el.currentTime)).boxes);
    } catch {
      setBoxes([]);
    }
  }, [video, textOnly]);

  function seek(t: number | null | undefined) {
    const el = videoEl.current;
    if (el && t != null) {
      el.pause();
      el.currentTime = t;
    }
  }

  function apply(reply: Reply) {
    if (reply.confirmation) setConfirmation(reply.confirmation);
    setQuestion(reply.question);
    setSpec(reply.spec);
    setCanApprove(reply.can_approve);
    say("assistant", reply.text, {
      question: reply.question ?? undefined,
      tone: reply.status === "error" ? "error" : undefined,
      spec: !!reply.spec && !reply.question,
    });
    if (reply.question) {
      seek(reply.question.seek_t ?? reply.confirmation?.keyframe_t);
      if (reply.question.operation_index != null) setSelectedRow(reply.question.operation_index);
    }
  }

  /** Run one request, show progress, and turn failures into a chat message. */
  async function run<T>(label: string, action: () => Promise<T>): Promise<T | null> {
    setBusy(label);
    try {
      return await action();
    } catch (error) {
      const text = error instanceof ApiError ? error.message : "알 수 없는 오류가 발생했습니다.";
      say("assistant", text, { tone: "error" });
      return null;
    } finally {
      setBusy(null);
    }
  }

  async function select(file: File) {
    cancelSelection();
    setView("original");
    setConfirmation(null); setQuestion(null); setSpec(null); setCanApprove(false); setRender(null);
    say("user", `영상 업로드: ${file.name}`);
    const uploaded = await run("영상 분석 중… (20초 영상 기준 약 20~30초)", () => api.upload(file));
    if (!uploaded) return;
    const session = await run("대화 준비 중…", () => api.createSession(uploaded.id));
    if (!session) return;
    setVideo(uploaded);
    setSessionId(session.id);
    setProvider(session.provider);
    const people = uploaded.tracks.filter(t => t.kind === "person").length;
    say("assistant", `분석이 끝났습니다. 사람 ${people}명을 찾았습니다. 편집 계획을 말씀해 주세요.`);
  }

  async function send() {
    const text = input.trim();
    if (!text || !sessionId || busy) return;
    setInput("");
    say("user", text);
    const el = videoEl.current;
    const reply = await run("AI가 계획을 해석하는 중…", () =>
      api.say(sessionId, text, { playhead: el ? el.currentTime : null, selection: null }),
    );
    if (reply) apply(reply);
  }

  async function answer(q: Question, option: string, label: string) {
    if (!sessionId || busy) return;
    say("user", label);
    const reply = await run("반영 중…", () => api.answer(sessionId, q.id, option));
    if (reply) apply(reply);
  }

  async function clickBox(trackId: string) {
    if (!sessionId || busy) return;
    if (selecting) {
      setInput(old => `${old}${old && !old.endsWith(" ") ? " " : ""}${trackId} `);
      cancelSelection();
      return;
    }
    const q = question;
    const index = q?.operation_index ?? (confirmation?.plan.operations.length ? selectedRow : null);
    if (index == null) {
      setInput(old => `${old}${trackId} `);
      return;
    }
    say("user", `[화면에서 ${trackId} 선택]`);
    const reply = await run("반영 중…", async () => {
      if (q?.kind === "choose") return api.choose(sessionId, index, [trackId]);
      if (q?.kind === "confirm") {
        if (q.candidates.includes(trackId)) return api.answer(sessionId, q.id, "yes");
        await api.answer(sessionId, q.id, "no");
        return api.choose(sessionId, index, [trackId]);
      }
      return api.toggle(sessionId, index, trackId);
    });
    if (reply) apply(reply);
  }

  async function removeRow(index: number) {
    if (!sessionId || busy) return;
    say("user", `[${index + 1}번 작업 삭제]`);
    const reply = await run("반영 중…", () => api.remove(sessionId, index));
    if (reply) { apply(reply); setSelectedRow(0); }
  }

  async function approve(preview: boolean) {
    if (!sessionId || busy) return;
    const result = await run(preview ? "미리보기 렌더링 중…" : "원본 해상도로 렌더링 중…", () =>
      api.approve(sessionId, preview),
    );
    if (!result) return;
    setRender(result.render);
    if (preview) {
      setView("complete");
      say("assistant", `명세서대로 렌더링했습니다 (${result.render.width}×${result.render.height}). '완료' 탭에서 확인하세요.`);
    } else {
      window.open(fileUrl(result.render.url), "_blank", "noopener");
      say("assistant", `원본 해상도 결과를 만들었습니다 (${result.render.width}×${result.render.height}).`);
    }
  }

  function add() {
    if (promptTimer.current) clearTimeout(promptTimer.current);
    setSelecting(true);
    setSelectPrompt(true);
    promptTimer.current = setTimeout(() => setSelectPrompt(false), 3000);
  }
  function cancelSelection() {
    if (promptTimer.current) clearTimeout(promptTimer.current);
    setSelecting(false);
    setSelectPrompt(false);
  }
  function changeView(next: "original" | "complete") {
    if (next === "complete") cancelSelection();
    setView(next);
  }

  // Colour each person box by what the plan does to them (or to their face).
  const boxViews: BoxView[] = useMemo(() => {
    if (!video) return [];
    const parentOf = new Map(video.tracks.map(t => [t.id, t.parent_id]));
    const role = new Map<string, BoxView["role"]>();
    for (const o of confirmation?.overlays ?? []) {
      // While "which one?" is open, the agent's provisional guess is not a decision: don't label it.
      if (question?.blocking && o.operation_index === question.operation_index && o.role === "target") continue;
      const owner = parentOf.get(o.track_id) ?? o.track_id;
      if (!role.has(owner) || o.role !== "target") role.set(owner, o.role);
    }
    const candidates = new Set(question?.candidates ?? []);
    return boxes
      .filter(b => b.kind === "person")
      .map(b => ({
        trackId: b.track_id,
        box: b.box,
        role: candidates.has(b.track_id)
          ? question?.kind === "confirm" ? "asked" : "candidate"
          : role.get(b.track_id) ?? "none",
      }));
  }, [boxes, confirmation, question, video]);

  const rows = rowsOf(confirmation);
  const lastQuestionMessage = [...messages].reverse().find(m => m.question)?.id;
  const lastAssistant = [...messages].reverse().find(m => m.sender === "assistant");
  const specMessage = spec && !question && lastAssistant?.spec ? lastAssistant.id : undefined;

  return (
    <main>
      <EditorHeader
        provider={provider}
        canExport={canApprove && !busy}
        onExport={() => approve(false)}
        textOnly={textOnly}
      />
      <div className="workspace">
        <div className={`editor ${view === "complete" ? "editor-complete" : ""}`}>
          <VideoPreview
            videoUrl={video ? fileUrl(video.proxy_url) : null}
            renderUrl={render ? fileUrl(render.url) : null}
            onSelect={select}
            selectPrompt={selectPrompt}
            selecting={selecting}
            onCancelSelection={cancelSelection}
            view={view}
            onViewChange={changeView}
            videoRef={videoEl}
            boxes={textOnly ? [] : boxViews}
            onBoxClick={clickBox}
            onPaused={refreshBoxes}
            onPlaying={() => setBoxes([])}
            busy={busy}
          />
          <WorkList
            works={rows}
            onAdd={add}
            onRemove={removeRow}
            readOnly={view === "complete" || !sessionId}
            selected={selectedRow}
            onSelectRow={setSelectedRow}
            spec={spec}
            canApprove={canApprove && !busy}
            onApprove={() => approve(true)}
            warnings={confirmation?.warnings ?? []}
          />
        </div>
        <ChatPanel
          messages={messages}
          input={input}
          setInput={setInput}
          onSend={send}
          disabled={!sessionId || !!busy}
          busy={busy}
          openQuestionMessage={question ? lastQuestionMessage : undefined}
          onAnswer={answer}
          specMessage={specMessage}
          canApprove={canApprove && !busy}
          onApprove={() => approve(true)}
        />
      </div>
    </main>
  );
}
