# LangGraph 0 to Hero — Part 2: Pro Topics

These lessons assume you finished `../lessons/` (especially lessons 3, 4 and 5).
Each file is a focused, runnable demo of one production pattern.

| # | Lesson | The problem it solves |
|---|--------|----------------------|
| 1 | `01_command_api.py` | Routing + updating in one place. A node returns `Command(goto=..., update=...)` instead of a plain dict + separate router — cleaner when the node itself knows where to go next. |
| 2 | `02_time_travel.py` | Every step is checkpointed. `get_state_history()` walks the timeline; resuming from an old checkpoint on a new thread *branches the past* — the basis of undo, auditing and replay. |
| 3 | `03_parallel_nodes.py` | Fan-out / fan-in. Edges from one node to many run in parallel; shared keys need a reducer (`operator.add`) or parallel writes silently overwrite each other. Map-reduce inside a graph. |
| 4 | `04_async_graphs.py` | Serving many users from one process: `async def` nodes, `await app.ainvoke(...)`, `async for ... in app.astream(...)`. |
| 5 | `05_error_handling.py` | Three failure modes: tools failing (error-as-text so the model can retry — note: modern `create_agent` does NOT auto-catch tool errors, unlike old `create_react_agent`), LLM failures (`.with_retry`), runaway loops (`recursion_limit` + `GraphRecursionError`). |
| 6 | `06_modern_create_agent.py` | The modern agent API: `from langchain.agents import create_agent` — the replacement for the deprecated `create_react_agent`, same engine, plus middleware hooks. |

## The modern API migration (important)

`create_react_agent` (used in `lessons/04` and `12`, and older tutorials) is
deprecated and disappears in LangGraph 2.0. The successor:

```python
from langchain.agents import create_agent          # new home
agent = create_agent(llm, tools, system_prompt=...)  # prompt= -> system_prompt=
```

Everything else (invoke format, threads, checkpointers) is unchanged.
Main behavioral gotcha: tool exceptions are no longer auto-converted to
ToolMessages — decide your own error policy (see lesson 5).

## Not covered here (worth knowing)

- **MCP (Model Context Protocol)** — connecting external tool servers (stdio, HTTP, remote with auth). This repo ships a working connector in `modules/tools/mcp_tools.py`, gated by `MCP_ENABLED=true`; see the main README's chatbot section.
- **LangSmith tracing** — set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` in
  `.env` and every run becomes a traceable, inspectable timeline in the UI. No
  code changes needed; skipped here only because it needs another API key.
- **Middleware** — `create_agent`'s extension point for logging/guardrails/
  approval hooks around every model and tool call.
- **Durable execution & multi-agent orchestration** (supervisor beyond lesson 10,
  LangGraph Platform) — deployment-scale topics.

---
*Part of the [LangGraph 0 to Hero Course](../README.md) • CC BY-NC 4.0 — free to learn, modify, and share; selling or paid teaching forbidden.*
