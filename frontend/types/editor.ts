import type { ConfirmStatus, Question } from "@/types/api";

/** One row of the work list: one operation of the current plan. */
export type WorkItem = {
  index: number;
  object: string;
  time: string;
  action: string;
  status: ConfirmStatus;
  warnings: number;
  enabled: boolean;
  rendered: boolean;
};

export type Message = {
  id: number;
  sender: "assistant" | "user";
  text: string;
  /** The question this message asks, so its answer buttons can be shown while it is open. */
  question?: Question;
  tone?: "error";
  /** This message presents the final spec: show the approve button under it while it stands. */
  spec?: boolean;
};

/** Used by lib/offlineChat.ts (the video-free demo chat); the editor itself uses types/api.ts. */
export type OfflineConfirmation = {
  plan: { operations: Array<{ op: string }> };
  items: Array<{
    operation_index: number;
    field: "op" | "selector" | "time_range";
    label: string;
    status: "ok" | "confirm" | "choose";
  }>;
};
