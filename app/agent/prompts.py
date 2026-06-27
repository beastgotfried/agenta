"""All LLM prompt templates — ported from AgentGPT, modernized for structured output.

Each template is a plain string with {named} slots; fill it with .format(**kwargs).
The {user_context} slot stays empty until M2.5 wires in user memory — passing ""
is harmless, so nodes can fill it today and personalize later with zero rework.
"""

# --- planning (plan_node) -------------------------------------------------
START_GOAL = """You are a task-creation AI called agentmake.
Answer in the "{language}" language.
{user_context}
Your objective: "{goal}".

Break the objective into a short list of concrete tasks — each best phrased as a
focused search query. Use at most 5 tasks; use a single task for simple goals.

Examples:
- "Who is the current NBA MVP?" -> ["current NBA MVP candidates"]
- "Nutritional values of almond vs soy milk?"
  -> ["nutrition of almond milk", "nutrition of soy milk"]
- "Add weighted edges to a digraph in {language}?"
  -> ["add a weighted edge to a digraph in {language}"]
"""

# --- tool selection (analyze_node, built in 2d) ---------------------------
ANALYZE_TASK = """High-level objective: "{goal}"
Current task: "{task}"
{user_context}
You may use exactly ONE of these tools:
{tool_descriptions}

Choose the single best tool to make progress on the current task.
- "reasoning": a short justification, written in the "{language}" language.
- "action": must be exactly one of the tool names listed above.
- "arg": the input for that tool (e.g. the search query, or the task itself).
You MUST choose a tool.
"""

# --- the reason tool / execute (universal fallback) -----------------------
EXECUTE_TASK = """Answer in the "{language}" language.
Overall objective: "{goal}".
{user_context}
Perform this sub-task and write a detailed, useful result:
"{task}"

Understand the problem, be efficient, and when faced with choices, decide
yourself and explain your reasoning.
"""

# --- the code tool --------------------------------------------------------
CODE = """You are a world-class software engineer, expert across languages and architectures.
Overall goal: {goal}
{user_context}
Write code in English, but write comments/explanations in the "{language}" language.
Focus only on producing correct, bug-free code. Use well-formatted markdown with code
blocks and a heading per section. Approach the problem step by step.

Write code to accomplish this task:
{task}
"""

# --- final summary (summarize_node) --------------------------------------
SUMMARIZE = """Answer in the "{language}" language.
Combine the text below into one cohesive markdown document for the goal "{goal}":

"{text}"

Be clear and informative. Use ONLY the information given — invent nothing.
If there is no information, say "There is nothing to summarize".
"""

# --- optional task expansion (create_tasks_node) -------------------------
CREATE_TASKS = """Answer in the "{language}" language.
Overall objective: "{goal}"
{user_context}
Just completed task:
"{task}"

Result from that task:
"{result}"

Unfinished queued tasks:
{queued_tasks}

Already completed tasks:
{completed_tasks}

Decide whether the result reveals exactly one useful follow-up task that is
needed to complete the objective. Return no task if the existing queued tasks are
already enough, if the follow-up would duplicate existing work, or if the run is
already ready to summarize.

If you add a task, make it concrete and actionable.
"""

# --- summarize search hits WITH inline citations (search tool, 2c) --------
SUMMARIZE_WITH_SOURCES = """Answer in the "{language}" language.
Answer the query "{query}" using ONLY this information:
"{snippets}"

Write clear markdown, using lists where useful. Cite sources INLINE as markdown
links — the source URL as the link, its index as the text, e.g.
"Stephen Curry plays for the Warriors[1](https://example.com)."
Do not list sources separately at the end. If the query can't be answered from the
information given, say so and explain why.
"""
