export type WorkItem = { id: number; object: string; time: string; action: string };
export type Message = { id: number; sender: "assistant" | "user"; text: string };
export type OfflineConfirmation = {
  plan: { operations: Array<{ op: string }> };
  items: Array<{
    operation_index: number;
    field: "op" | "selector" | "time_range";
    label: string;
    status: "ok" | "confirm" | "choose";
  }>;
};
