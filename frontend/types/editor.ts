export type WorkItem = { id: number; object: string; time: string; action: string };
export type Message = { id: number; sender: "assistant" | "user"; text: string };
