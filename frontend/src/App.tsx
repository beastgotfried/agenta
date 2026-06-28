import {
  Activity,
  AlertCircle,
  Brain,
  CheckCircle2,
  Circle,
  Code2,
  Flag,
  History,
  MessageSquare,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Search,
  Send,
  X,
  Zap
} from "lucide-react";
import { type CSSProperties, FormEvent, useEffect, useRef, useState } from "react";
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

const NODE_SEQUENCE = [
  ["plan", "Plan"],
  ["pick_task", "Pick"],
  ["analyze", "Analyze"],
  ["execute", "Execute"],
  ["create_tasks", "Expand"],
  ["summarize", "Summarize"]
] as const;

const NODE_LABELS = Object.fromEntries(NODE_SEQUENCE) as Record<string, string>;

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

function nodeLabel(node: string): string {
  return NODE_LABELS[node] || node.replace(/_/g, " ");
}

function formatChatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit"
  }).format(date);
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
          <a href={part} key={`${part}-${index}`} target="_blank" rel="noreferrer">
            {part}
          </a>
        );
      })}
    </>
  );
}

function EmptyTrace() {
  return (
    <div className="empty-trace">
      <Radio aria-hidden="true" size={28} />
      <div>
        <strong>No Trace Yet</strong>
        <span>Start a run to populate the execution spine.</span>
      </div>
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
    if (!window.confirm("Cancel this run? The current graph step stops after it finishes.")) {
      return;
    }

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
  const completedCount = tasks.filter((task) => task.status === "completed").length;
  const queuedCount = tasks.filter((task) => task.status === "queued").length;
  const activeTask = tasks.find((task) => task.status === "active");
  const progress = tasks.length === 0 ? 0 : Math.round((completedCount / tasks.length) * 100);
  const lastEvent = events.length > 0 ? events[events.length - 1] : undefined;
  const activeTool = activeTask?.analysis?.action || "reason";
  const ActiveToolIcon = TOOL_ICONS[activeTool];
  const progressStyle = { "--progress": `${progress}%` } as CSSProperties;

  return (
    <>
      <a className="skip-link" href="#trace-title">
        Skip To Run Workspace
      </a>
      <main className="app-shell">
        <header className="topbar">
          <div className="brand-stack">
            <div className="brand-mark" aria-hidden="true">
              ag
            </div>
            <div>
              <p className="eyebrow">agentmake</p>
              <h1 translate="no">agenta</h1>
            </div>
          </div>
          <div className={`status-pill ${status}`} aria-live="polite">
            <span aria-hidden="true" />
            {STATUS_LABELS[status]}
          </div>
        </header>

        <section className="workspace" aria-label="Agenta run console">
          <aside className="launch-dock" aria-labelledby="launch-title">
            <div className="dock-heading">
              <p className="eyebrow">Launch</p>
              <h2 id="launch-title">New Run</h2>
            </div>

            <form onSubmit={handleStart} className="run-form">
              <label htmlFor="goal">Goal</label>
              <textarea
                autoComplete="off"
                id="goal"
                name="goal"
                value={goal}
                onChange={(event) => setGoal(event.target.value)}
                rows={8}
                placeholder="Research LangGraph checkpointing, compare patterns, then summarize the API ideas…"
              />

              <div className="field-grid">
                <label htmlFor="language">
                  Language
                  <input
                    autoComplete="off"
                    id="language"
                    name="language"
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                  />
                </label>
                <label htmlFor="max-loops">
                  Loops
                  <input
                    autoComplete="off"
                    id="max-loops"
                    inputMode="numeric"
                    max={25}
                    min={1}
                    name="max_loops"
                    type="number"
                    value={maxLoops}
                    onChange={(event) =>
                      setMaxLoops(Math.min(25, Math.max(1, Number(event.target.value) || 1)))
                    }
                  />
                </label>
              </div>

              <label className="switch-row">
                <input
                  checked={expandTasks}
                  name="expand_tasks"
                  onChange={(event) => setExpandTasks(event.target.checked)}
                  type="checkbox"
                />
                <span aria-hidden="true" />
                Expand Tasks
              </label>

              <button className="primary-button" disabled={!canStart} type="submit">
                <Play aria-hidden="true" size={17} />
                Start Run
              </button>
            </form>

            <div className="control-row" aria-label="Run controls">
              <button disabled={!canPause} onClick={handlePause} type="button">
                <Pause aria-hidden="true" size={16} />
                Pause
              </button>
              <button disabled={!canResume} onClick={handleResume} type="button">
                <RefreshCw aria-hidden="true" size={16} />
                Resume
              </button>
              <button
                className="danger-button"
                disabled={!canCancel}
                onClick={handleCancel}
                type="button"
              >
                <X aria-hidden="true" size={16} />
                Cancel
              </button>
            </div>

            <dl className="run-facts">
              <div>
                <dt>Backend</dt>
                <dd translate="no">{API_BASE}</dd>
              </div>
              <div>
                <dt>Run ID</dt>
                <dd translate="no">{runId || "none"}</dd>
              </div>
              <div>
                <dt>Node</dt>
                <dd translate="no">{nodeLabel(activeNode)}</dd>
              </div>
            </dl>

            <section className="tool-list" aria-labelledby="tools-title">
              <h2 id="tools-title">Tools</h2>
              <ul>
                {tools.map((tool) => {
                  const Icon = TOOL_ICONS[tool.name] || Brain;
                  return (
                    <li key={tool.name} title={tool.description}>
                      <Icon aria-hidden="true" size={16} />
                      <span translate="no">{tool.name}</span>
                    </li>
                  );
                })}
              </ul>
            </section>
          </aside>

          <section className="trace-desk" aria-labelledby="trace-title">
            {error ? (
              <div className="error-banner" role="alert">
                <AlertCircle aria-hidden="true" size={18} />
                {error}
              </div>
            ) : null}

            <div className="trace-hero">
              <div>
                <p className="eyebrow">Live Graph</p>
                <h2 id="trace-title">Execution Spine</h2>
              </div>
              <div className="run-signal" aria-label={`Progress ${progress}%`} style={progressStyle}>
                <span aria-hidden="true" />
                <strong>{progress}%</strong>
              </div>
            </div>

            <div className="metric-strip">
              <div>
                <span>Completed</span>
                <strong>
                  {completedCount}/{tasks.length || 0}
                </strong>
              </div>
              <div>
                <span>Queued</span>
                <strong>{queuedCount}</strong>
              </div>
              <div>
                <span>Events</span>
                <strong>{events.length}</strong>
              </div>
              <div>
                <span>Latest</span>
                <strong translate="no">{lastEvent?.type || "none"}</strong>
              </div>
            </div>

            <ol className="node-ribbon" aria-label="Graph node path">
              {NODE_SEQUENCE.map(([node, label]) => (
                <li className={activeNode === node ? "active" : ""} key={node}>
                  <span aria-hidden="true" />
                  {label}
                </li>
              ))}
            </ol>

            <div className="trace-focus">
              <ActiveToolIcon aria-hidden="true" size={18} />
              <span>Active Tool</span>
              <strong translate="no">{activeTask?.analysis?.action || "waiting"}</strong>
              <p>{activeTask?.title || "No task is executing right now."}</p>
            </div>

            {tasks.length === 0 ? (
              <EmptyTrace />
            ) : (
              <ol className="trace-list">
                {tasks.map((task, index) => {
                  const Icon = task.status === "completed" ? CheckCircle2 : Circle;
                  const ToolIcon = task.analysis ? TOOL_ICONS[task.analysis.action] : Brain;
                  return (
                    <li className={`trace-item ${task.status}`} key={task.id}>
                      <article aria-current={task.status === "active" ? "step" : undefined}>
                        <div className="task-index" aria-hidden="true">
                          {String(index + 1).padStart(2, "0")}
                        </div>
                        <div className="task-body">
                          <div className="task-title">
                            <Icon aria-hidden="true" size={18} />
                            <h3>{task.title}</h3>
                            <span className="task-state">{task.status}</span>
                          </div>

                          {task.analysis ? (
                            <div className="analysis-row">
                              <ToolIcon aria-hidden="true" size={16} />
                              <span translate="no">{task.analysis.action}</span>
                              <p>{task.analysis.reasoning}</p>
                            </div>
                          ) : null}

                          {task.result ? (
                            <div className="result-block">
                              <LinkifiedText text={task.result} />
                            </div>
                          ) : null}
                        </div>
                      </article>
                    </li>
                  );
                })}
              </ol>
            )}

            <section className="summary-panel" aria-live="polite">
              <div className="section-heading compact">
                <div>
                  <p className="eyebrow">Output</p>
                  <h2>Summary</h2>
                </div>
                {status === "running" ? <Activity aria-hidden="true" size={16} /> : null}
              </div>
              <div className="summary-body">
                {summary ? <LinkifiedText text={summary} /> : <span>No summary yet.</span>}
              </div>
            </section>
          </section>

          <aside className="debrief-dock" aria-labelledby="chat-title">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Debrief</p>
                <h2 id="chat-title">Run Chat</h2>
              </div>
              <button
                aria-label="Load chat history"
                className="icon-button"
                disabled={!canChat}
                onClick={loadChatHistory}
                title="Load chat history"
                type="button"
              >
                <History aria-hidden="true" size={16} />
              </button>
            </div>

            <div className="chat-thread" aria-live="polite" role="log">
              {chatMessages.length === 0 ? (
                <div className="chat-empty">
                  <MessageSquare aria-hidden="true" size={20} />
                  <span>No Messages Yet</span>
                </div>
              ) : (
                chatMessages.map((message) => {
                  const time = formatChatTime(message.created_at);
                  return (
                    <article className="chat-message" key={message.id}>
                      <div>
                        <strong>{message.question}</strong>
                        {time ? (
                          <time dateTime={message.created_at}>{time}</time>
                        ) : null}
                      </div>
                      <p>
                        <LinkifiedText text={message.answer} />
                      </p>
                    </article>
                  );
                })
              )}
            </div>

            <form className="chat-form" onSubmit={handleChat}>
              <label htmlFor="chat-question">Question</label>
              <textarea
                autoComplete="off"
                disabled={!canChat || isChatLoading}
                id="chat-question"
                name="chat_question"
                onChange={(event) => setChatQuestion(event.target.value)}
                placeholder="Ask what the run concluded…"
                rows={4}
                value={chatQuestion}
              />
              <button disabled={!canChat || isChatLoading || !chatQuestion.trim()} type="submit">
                {isChatLoading ? (
                  <Zap aria-hidden="true" size={16} />
                ) : (
                  <Send aria-hidden="true" size={16} />
                )}
                {isChatLoading ? "Asking…" : "Ask Run"}
              </button>
            </form>
          </aside>
        </section>
      </main>
    </>
  );
}
