// Thin client for the CutPilot backend (docs/api.md). Every editing call returns a Reply,
// so the editor redraws from the latest Reply alone.
import type { Render, Reply, Session, TrackBox, Video } from "@/types/api";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

/** Server paths such as /files/... become absolute URLs. */
export const fileUrl = (path: string) => `${API_BASE}${path}`;

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, `서버에 연결할 수 없습니다 (${API_BASE}). 백엔드가 켜져 있는지 확인해 주세요.`);
  }
  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = (await res.json()).detail;
    } catch {
      /* not JSON */
    }
    const reasons = (detail as { reasons?: string[] } | null)?.reasons;
    const message = typeof detail === "string" ? detail : reasons?.join("\n") ?? res.statusText;
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}

const send = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export type UIState = { playhead: number | null; selection: { start: number; end: number } | null };

export const api = {
  upload(file: File) {
    const form = new FormData();
    form.append("file", file);
    return call<Video>("/videos", { method: "POST", body: form });
  },
  boxes(videoId: string, t: number) {
    return call<{ t: number; boxes: TrackBox[] }>(`/videos/${videoId}/boxes?t=${t.toFixed(2)}`);
  },
  createSession(videoId: string) {
    return call<Session>("/sessions", send("POST", { video_id: videoId }));
  },
  say(sessionId: string, text: string, ui: UIState) {
    return call<Reply>(`/sessions/${sessionId}/messages`, send("POST", { text, ui }));
  },
  answer(sessionId: string, questionId: string, option: string) {
    return call<Reply>(`/sessions/${sessionId}/questions/${questionId}/answer`, send("POST", { option }));
  },
  choose(sessionId: string, index: number, trackIds: string[]) {
    return call<Reply>(`/sessions/${sessionId}/operations/${index}/choose`, send("POST", { track_ids: trackIds }));
  },
  toggle(sessionId: string, index: number, trackId: string) {
    return call<Reply>(`/sessions/${sessionId}/operations/${index}/toggle`, send("POST", { track_id: trackId }));
  },
  remove(sessionId: string, index: number) {
    return call<Reply>(`/sessions/${sessionId}/operations/${index}`, send("DELETE"));
  },
  approve(sessionId: string, preview: boolean) {
    return call<{ render: Render }>(`/sessions/${sessionId}/approve`, send("POST", { preview }));
  },
};
