import type {
  ChatMessage,
  CreateRunRequest,
  CreateRunResponse,
  RunStatusResponse,
  StreamEvent,
  ToolInfo
} from "./types";

export const API_BASE =
  import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers
    }
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP status text when the response is not JSON.
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export function getTools(): Promise<ToolInfo[]> {
  return request<ToolInfo[]>("/tools");
}

export function createRun(payload: CreateRunRequest): Promise<CreateRunResponse> {
  return request<CreateRunResponse>("/runs", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function pauseRun(runId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/runs/${runId}/pause`, { method: "POST" });
}

export function resumeRun(runId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/runs/${runId}/resume`, { method: "POST" });
}

export function cancelRun(runId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/runs/${runId}/cancel`, { method: "POST" });
}

export function listChat(runId: string): Promise<ChatMessage[]> {
  return request<ChatMessage[]>(`/runs/${runId}/chat`);
}

export function askRun(runId: string, question: string): Promise<ChatMessage> {
  return request<ChatMessage>(`/runs/${runId}/chat`, {
    method: "POST",
    body: JSON.stringify({ question })
  });
}

export function openRunStream(
  runId: string,
  onEvent: (event: StreamEvent) => void,
  onError: (message: string) => void
): EventSource {
  const source = new EventSource(`${API_BASE}/runs/${runId}/stream`);
  const eventTypes = [
    "status",
    "node",
    "plan",
    "task",
    "analysis",
    "task_done",
    "task_created",
    "summary",
    "error",
    "done"
  ];

  for (const eventType of eventTypes) {
    source.addEventListener(eventType, (message) => {
      try {
        onEvent(JSON.parse(message.data) as StreamEvent);
      } catch {
        onError("Received an unreadable stream event.");
      }
    });
  }

  source.onerror = () => {
    onError("The stream connection closed unexpectedly.");
    source.close();
  };

  return source;
}
