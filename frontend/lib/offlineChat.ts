import type { OfflineConfirmation } from "@/types/editor";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export class OfflineChatError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
  }
}

async function post<T>(path: string, body?: object): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new OfflineChatError("백엔드 서버에 연결할 수 없습니다. 서버 실행 상태를 확인해 주세요.");
  }
  if (!response.ok) {
    throw new OfflineChatError(
      response.status === 404
        ? "오프라인 대화가 종료되었습니다. 다시 보내면 새 대화가 시작됩니다."
        : "오프라인 응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요.",
      response.status,
    );
  }
  return response.json() as Promise<T>;
}

export function createOfflineSession() {
  return post<{ id: string; mode: "offline-demo" }>("/demo/offline-sessions");
}

export function sendOfflineMessage(sessionId: string, text: string) {
  return post<{ status: "plan" | "clarify" | "error"; text: string; blockers: string[]; confirmation: OfflineConfirmation | null }>(
    `/demo/offline-sessions/${encodeURIComponent(sessionId)}/messages`,
    { text },
  );
}
