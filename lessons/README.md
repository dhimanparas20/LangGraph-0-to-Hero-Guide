# LangGraph 0 to Hero — Part 1: The Lessons

This folder teaches LangGraph from absolute zero, one runnable file per lesson.
Every file is self-contained and prints a trace of what happens — read the code,
run it, and match the prints to the concepts.

**Prerequisites:** you know basic Python. Knowing LangChain (models, prompts,
messages) helps but is not required.

**How to run** (from the repo root, in WSL):

```bash
uv run lessons/01_first_graph.py
```

You need a `.env` file (copy `.env.sample`) with an `OPENAI_API_KEY`.
Lessons 1 and up are ordered — each one assumes the previous.

## The path

| # | Lesson | What you learn |
|---|--------|----------------|
| 1 | `01_first_graph.py` | **The graph core.** State (shared data), nodes (functions that return only what changed), edges (the wiring), `compile()`, `invoke()`. Run it before anything else. |
| 2 | `02_llm_node.py` | **The LLM lives in nodes.** A node calls ChatOpenAI like any code. The `messages` key + `add_messages` reducer makes history *append* instead of overwrite — this is chat memory. |
| 3 | `03_conditional_edges.py` | **The graph makes decisions.** Router functions pick the next node at runtime; edges can point backwards = loops; every loop needs a stopping condition. Demo: a bot that rewrites its too-long answer. |
| 4 | `04_tool_agent.py` | **Agents.** Tools are functions with docstrings; `bind_tools`; the model requests calls via `tool_calls`; Part 2 rebuilds it with the prebuilt `ToolNode`. |
| 5 | `05_memory.py` | **Memory across turns.** A checkpointer saves state after every step; `thread_id` = one conversation; you send only the new message. |
| 6 | `06_rag.py` | **RAG is just nodes.** A retrieve node searches Chroma, a generate node answers. Two state keys with two merge rules (append vs replace). |
| 7 | `07_adaptive_rag.py` | **The graph decides how to answer.** Router (direct vs retrieve), LLM grader for doc relevance, query-rewrite retry loop, graceful "I don't know". |
| 8 | `08_human_in_loop.py` | **Pause for approval.** `interrupt_before` stops before a node; approve with `invoke(None)`, reject with `update_state(as_node=...)`. |
| 9 | `09_streaming.py` | **Watch it work live.** `stream()` modes: `updates` (which node ran), `values` (full state), `messages` (tokens). |
| 10 | `10_subgraphs.py` | **Graphs inside graphs.** A compiled graph as a node; supervisor pattern routing to specialist bots — the seed of multi-agent systems. |
| 11 | `11_sqlite_memory.py` | **Memory that survives restarts.** `SqliteSaver` writes checkpoints to a file. Run this file twice — the second run remembers the first! |
| 12 | `12_prebuilt_agent.py` | **The one-liner.** `create_react_agent` is everything you hand-built, prebuilt — with RAG as a *tool* and memory on top. |

## Conventions used in every lesson

- State schemas are **Pydantic** models (defaults + validation)
- Nodes print `node_name:` lines so output reads like an execution log
- Model names come from `.env`, never hardcoded
- The Python stays simple on purpose — the complexity belongs to LangGraph

After finishing: `../pro_topics/` for production patterns, `../examples/` for
full mini-projects, and the repo-root `README.md` for a complete revision guide.

---
*Part of the [LangGraph 0 to Hero Course](../README.md) • CC BY-NC 4.0 — free to learn, modify, and share; selling or paid teaching forbidden.*
