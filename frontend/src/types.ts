export type RunStatus =
  | "idle"
  | "created"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "cancelled";

export type StreamType =
  | "status"
  | "node"
  | "plan"
  | "task"
  | "analysis"
  | "task_done"
  | "task_created"
  | "summary"
  | "error"
  | "done";

export type ToolName = "reason" | "search" | "code" | "conclude";

export interface ToolInfo {
  name: ToolName;
  description: string;
  arg_description: string;
}

export interface CreateRunRequest {
  goal: string;
  language?: string;
  max_loops?: number;
  expand_tasks: boolean;
}

export interface CreateRunResponse {
  run_id: string;
  status: Exclude<RunStatus, "idle">;
}

export interface RunStatusResponse {
  run_id: string;
  status: Exclude<RunStatus, "idle">;
}

export interface AnalysisPayload {
  task: string | null;
  analysis: {
    reasoning: string;
    action: ToolName;
    arg: string;
  };
}

export interface StreamEvent<TPayload = Record<string, unknown>> {
  run_id: string;
  sequence: number;
  type: StreamType;
  payload: TPayload;
}

export interface TaskView {
  id: string;
  title: string;
  status: "queued" | "active" | "completed" | "created";
  analysis?: AnalysisPayload["analysis"];
  result?: string;
  loopCount?: number;
}

export interface ChatMessage {
  id: number;
  run_id: string;
  question: string;
  answer: string;
  created_at: string;
}
