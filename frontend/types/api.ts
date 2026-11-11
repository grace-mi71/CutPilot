// Shapes returned by the CutPilot backend. See docs/api.md for what each field means.

export type Box = { x: number; y: number; w: number; h: number };
export type TimeRange = { start: number | null; end: number | null };

export type Track = {
  id: string;
  kind: "person" | "face" | "license_plate";
  parent_id: string | null;
  display_name: string | null;
  segments: { start: number; end: number }[];
};

export type Video = {
  id: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  proxy_url: string;
  sprite_url: string;
  sprite_interval: number;
  sprite_columns: number;
  tracks: Track[];
};

export type TrackBox = { track_id: string; kind: Track["kind"]; box: Box };

export type OverlayRole = "target" | "candidate" | "excluded" | "reframe_subject";
export type Overlay = {
  track_id: string;
  box: Box;
  role: OverlayRole;
  label: string | null;
  operation_index: number | null;
};

export type ConfirmStatus = "ok" | "confirm" | "choose";
export type ConfirmItem = {
  operation_index: number;
  field: "op" | "selector" | "time_range" | "region";
  label: string;
  status: ConfirmStatus;
  candidates: string[];
};

export type PlanWarning = {
  kind: "track_gap" | "out_of_frame" | "empty_selection";
  message: string;
  operation_index: number | null;
  track_id: string | null;
  time_range: TimeRange | null;
};

export type Operation = {
  op: "MOSAIC" | "BLUR" | "REFRAME";
  selector: { kind: Track["kind"]; include: string[]; exclude: string[] };
  time_range: TimeRange;
  aspect_ratio: string | null;
  strength: number | null;
  enabled: boolean;
};

export type Confirmation = {
  plan: { video_id: string; operations: Operation[] };
  keyframe_t: number;
  overlays: Overlay[];
  items: ConfirmItem[];
  warnings: PlanWarning[];
};

export type Question = {
  id: string;
  kind: "choose" | "missing" | "confirm" | "track_gap" | "out_of_frame";
  operation_index: number | null;
  text: string;
  blocking: boolean;
  options: { id: string; label: string }[];
  candidates: string[];
  track_id: string | null;
  time_range: TimeRange | null;
  seek_t: number | null;
};

export type Spec = {
  steps: { index: number; op: string; label: string; targets: string[]; excluded: string[]; time: string }[];
  decisions: string[];
  text: string;
};

export type Reply = {
  status: "plan" | "clarify" | "error";
  text: string;
  notes: string[];
  confirmation: Confirmation | null;
  question: Question | null;
  remaining_questions: number;
  spec: Spec | null;
  can_approve: boolean;
  blockers: string[];
  rendered_indices: number[];
};

export type Render = { url: string; preview: boolean; width: number; height: number; frames: number };

export type Session = {
  id: string;
  video_id: string;
  provider: string;
  confirmation: Confirmation | null;
  question: Question | null;
  questions: Question[];
  spec: Spec | null;
  decisions: string[];
  can_approve: boolean;
  blockers: string[];
  history: { role: string; text: string }[];
  render: Render | null;
  rendered_indices: number[];
};
