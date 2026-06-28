import {
  AlertCircle,
  Brain,
  CheckCircle2,
  Circle,
  Code2,
  Flag,
  MessageSquare,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Search,
  Send,
  Square,
  X
} from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  askRun,
  cancelRun,
  createRun,
  getTools,
  listChat,
  openRunStream,
  pauseRun,
  resumeRun
} from "./api";
import type {
  AnalysisPayload,
  ChatMessage,
  RunStatus,
  StreamEvent,
  TaskView,
  ToolInfo,
  ToolName
} from "./types";

const STATUS_LABELS: Record<RunStatus, string> = {
  idle: "Idle",
  created: "Created",
  running: "Running",
  paused: "Paused",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled"
};

const TOOL_ICONS: Record<ToolName, typeof Brain> = {
  reason: Brain,
  search: Search,
  code: Code2,
  conclude: Flag
};

function taskId(title: string): string {
  return title.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "");
}

function updateTask(
  tasks: TaskView[],
  title: string,
  update: Partial<TaskView>
): TaskView[] {
  const id = taskId(title);
  const existing = tasks.find((task) => task.id === id);

  if (!existing) {
    return [...tasks, { id, title, status: "queued", ...update }];
  }

  return tasks.map((task) => (task.id === id ? { ...task, ...update } : task));
}

function LinkifiedText({ text }: { text: string }) {
  const parts = text.split(/(https?:\/\/[^\s)]+)/g);

  return (
    <>
      {parts.map((part, index) => {
        if (!/^https?:\/\//.test(part)) {
          return <span key={`${part}-${index}`}>{part}</span>;
        }

        return (
          <a href={part} key={part} target="_blank" rel="noreferrer">
            {part}
          </a>
        );
      })}
    </>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <Radio size={26} />
      <span>No stream events yet.</span>
    </div>
  );
}

export default function App() {
  const [goal, setGoal] = useState("");
  const [language, setLanguage] = useState("English");
  const [maxLoops, setMaxLoops] = useState(8);
  const [expandTasks, setExpandTasks] = useState(false);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [tasks, setTasks] = useState<TaskView[]>([]);
  const [summary, setSummary] = useState("");
  const [activeNode, setActiveNode] = useState("none");
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [error, setError] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatQuestion, setChatQuestion] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const streamRef = useRef<EventSource | null>(null);

  useEffect(() => {
    getTools()
      .then(setTools)
      .catch((caught: Error) => setError(caught.message));

    return () => {
      streamRef.current?.close();
    };
  }, []);

  function closeStream() {
    streamRef.current?.close();
    streamRef.current = null;
  }

  function handleStreamEvent(event: StreamEvent) {
    setEvents((current) => [...current, event]);

    if (event.type === "status") {
      const nextStatus = (event.payload as { status?: RunStatus }).status;
      if (nextStatus) {
        setStatus(nextStatus);
      }
      return;
    }

    if (event.type === "node") {
      setActiveNode((event.payload as { node?: string }).node || "unknown");
      return;
    }

    if (event.type === "plan") {
      const plannedTasks = ((event.payload as { tasks?: string[] }).tasks || []).map(
        (title) => ({ id: taskId(title), title, status: "queued" as const })
      );
      setTasks(plannedTasks);
      return;
    }

    if (event.type === "task") {
      const title = (event.payload as { task?: string }).task;
      if (title) {
        setTasks((current) => updateTask(current, title, { status: "active" }));
      }
      return;
    }

    if (event.type === "analysis") {
      const payload = event.payload as unknown as AnalysisPayload;
      if (payload.task) {
        setTasks((current) =>
          updateTask(current, payload.task as string, {
            status: "active",
            analysis: payload.analysis
          })
        );
      }
      return;
    }

    if (event.type === "task_done") {
      const payload = event.payload as {
        task?: string;
        result?: string;
        loop_count?: number;
      };
      if (payload.task) {
        setTasks((current) =>
          updateTask(current, payload.task as string, {
            status: "completed",
            result: payload.result || "",
            loopCount: payload.loop_count
          })
        );
      }
      return;
    }

    if (event.type === "task_created") {
      const title = (event.payload as { task?: string }).task;
      if (title) {
        setTasks((current) => updateTask(current, title, { status: "created" }));
      }
      return;
    }

    if (event.type === "summary") {
      setSummary((event.payload as { text?: string }).text || "");
      return;
    }

    if (event.type === "error") {
      setError((event.payload as { message?: string }).message || "Run failed.");
      setStatus("failed");
      closeStream();
      return;
    }

    if (event.type === "done") {
      setStatus((current) =>
        current === "running" || current === "created" ? "completed" : current
      );
      closeStream();
    }
  }

  function connectStream(nextRunId: string) {
    closeStream();
    streamRef.current = openRunStream(
      nextRunId,
      handleStreamEvent,
      (message) => {
        setError(message);
        closeStream();
      }
    );
  }

  async function handleStart(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedGoal = goal.trim();
    if (!trimmedGoal) {
      setError("Enter a goal first.");
      return;
    }

    closeStream();
    setError("");
    setSummary("");
    setTasks([]);
    setEvents([]);
    setChatMessages([]);
    setActiveNode("none");
    setStatus("created");

    try {
      const run = await createRun({
        goal: trimmedGoal,
        language: language.trim() || "English",
        max_loops: maxLoops,
        expand_tasks: expandTasks
      });
      setRunId(run.run_id);
      setStatus(run.status);
      connectStream(run.run_id);
    } catch (caught) {
      setStatus("failed");
      setError(caught instanceof Error ? caught.message : "Could not start run.");
    }
  }

  async function handlePause() {
    if (!runId) return;
    try {
      const response = await pauseRun(runId);
      setStatus(response.status);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not pause run.");
    }
  }

  async function handleResume() {
    if (!runId) return;
    try {
      const response = await resumeRun(runId);
      setStatus(response.status);
      connectStream(runId);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not resume run.");
    }
  }

  async function handleCancel() {
    if (!runId) return;
    try {
      const response = await cancelRun(runId);
      setStatus(response.status);
      closeStream();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not cancel run.");
    }
  }

  async function handleChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!runId || !chatQuestion.trim()) return;

    setIsChatLoading(true);
    setError("");
    try {
      const message = await askRun(runId, chatQuestion.trim());
      setChatMessages((current) => [...current, message]);
      setChatQuestion("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Chat request failed.");
    } finally {
      setIsChatLoading(false);
    }
  }

  async function loadChatHistory() {
    if (!runId) return;
    try {
      setChatMessages(await listChat(runId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load chat history.");
    }
  }

  const canStart = status !== "running" && goal.trim().length > 0;
  const canPause = status === "running" && Boolean(runId);
  const canResume = status === "paused" && Boolean(runId);
  const canCancel = ["created", "running", "paused"].includes(status) && Boolean(runId);
  const canChat = status === "completed" && Boolean(runId);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">agentmake</p>
          <h1>agenta</h1>
        </div>
        <div className={`status-pill ${status}`}>
          <span />
          {STATUS_LABELS[status]}
        </div>
      </header>

      <section className="workspace">
        <aside className="control-panel">
          <form onSubmit={handleStart} className="run-form">
            <label htmlFor="goal">Goal</label>
            <textarea
              id="goal"
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              rows={7}
              placeholder="Research LangGraph checkpointing and summarize the important API ideas"
            />

            <div className="field-grid">
              <label>
                Language
                <input
                  value={language}
                  onChange={(event) => setLanguage(event.target.value)}
                />
              </label>
              <label>
                Loops
                <input
                  min={1}
                  max={25}
                  type="number"
                  value={maxLoops}
                  onChange={(event) =>
                    setMaxLoops(Math.min(25, Math.max(1, Number(event.target.value) || 1)))
                  }
                />
              </label>
            </div>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={expandTasks}
                onChange={(event) => setExpandTasks(event.target.checked)}
              />
              Expand tasks
            </label>

            <button className="primary-button" disabled={!canStart} type="submit">
              <Play size={17} />
              Start run
            </button>
          </form>

          <div className="control-row">
            <button disabled={!canPause} onClick={handlePause} title="Pause run">
              <Pause size={16} />
              Pause
            </button>
            <button disabled={!canResume} onClick={handleResume} title="Resume run">
              <RefreshCw size={16} />
              Resume
            </button>
            <button disabled={!canCancel} onClick={handleCancel} title="Cancel run">
              <X size={16} />
              Cancel
            </button>
          </div>

          <div className="meta-panel">
            <div>
              <span>Backend</span>
              <strong>{API_BASE}</strong>
            </div>
            <div>
              <span>Run ID</span>
              <strong>{runId || "none"}</strong>
            </div>
            <div>
              <span>Node</span>
              <strong>{activeNode}</strong>
            </div>
          </div>

          <div className="tool-list">
            <h2>Tools</h2>
            {tools.map((tool) => {
              const Icon = TOOL_ICONS[tool.name] || Brain;
              return (
                <div className="tool-row" key={tool.name}>
                  <Icon size={16} />
                  <span>{tool.name}</span>
                </div>
              );
            })}
          </div>
        </aside>

        <section className="run-panel">
          {error ? (
            <div className="error-banner">
              <AlertCircle size={18} />
              {error}
            </div>
          ) : null}

          <div className="section-heading">
            <div>
              <p className="eyebrow">Run</p>
              <h2>Task stream</h2>
            </div>
            <span>{events.length} events</span>
          </div>

          {tasks.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="task-list">
              {tasks.map((task) => {
                const Icon = task.status === "completed" ? CheckCircle2 : Circle;
                const ToolIcon = task.analysis ? TOOL_ICONS[task.analysis.action] : Brain;
                return (
                  <article className={`task-card ${task.status}`} key={task.id}>
                    <div className="task-title">
                      <Icon size={18} />
                      <h3>{task.title}</h3>
                    </div>

                    {task.analysis ? (
                      <div className="analysis-row">
                        <ToolIcon size={16} />
                        <span>{task.analysis.action}</span>
                        <p>{task.analysis.reasoning}</p>
                      </div>
                    ) : null}

                    {task.result ? (
                      <div className="result-block">
                        <LinkifiedText text={task.result} />
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          )}

          <div className="summary-panel">
            <div className="section-heading compact">
              <div>
                <p className="eyebrow">Output</p>
                <h2>Summary</h2>
              </div>
              {status === "running" ? <Square size={14} /> : null}
            </div>
            <div className="summary-body">
              {summary ? <LinkifiedText text={summary} /> : <span>No summary yet.</span>}
            </div>
          </div>
        </section>

        <aside className="chat-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Completed run</p>
              <h2>Chat</h2>
            </div>
            <button disabled={!canChat} onClick={loadChatHistory} title="Load chat history">
              <MessageSquare size={16} />
            </button>
          </div>

          <div className="chat-thread">
            {chatMessages.length === 0 ? (
              <span>No messages yet.</span>
            ) : (
              chatMessages.map((message) => (
                <div className="chat-message" key={message.id}>
                  <strong>{message.question}</strong>
                  <p>
                    <LinkifiedText text={message.answer} />
                  </p>
                </div>
              ))
            )}
          </div>

          <form className="chat-form" onSubmit={handleChat}>
            <textarea
              disabled={!canChat || isChatLoading}
              value={chatQuestion}
              onChange={(event) => setChatQuestion(event.target.value)}
              placeholder="Ask about the completed run"
              rows={3}
            />
            <button disabled={!canChat || isChatLoading || !chatQuestion.trim()} type="submit">
              <Send size={16} />
              Send
            </button>
          </form>
        </aside>
      </section>
    </main>
  );
}
