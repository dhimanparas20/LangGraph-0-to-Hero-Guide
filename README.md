# LangGraph 0 to Hero — The Complete Course

A full 0-to-hero LangGraph course: **12 fundamental lessons → 6 pro topics → 4 real-life example projects → 1 production-style CLI chatbot**, plus this file as your complete revision guide.

📖 **License: free to read, run, modify, share, and contribute — selling it or teaching it for profit is forbidden.** See [LICENSE](LICENSE) (CC BY-NC 4.0).

## Repo map

| Folder | What's inside | Read its README |
|---|---|---|
| `lessons/` | Part 1 — the 0-to-hero course: 12 ordered, runnable lessons | [lessons/README.md](lessons/README.md) |
| `pro_topics/` | Part 2 — 6 production patterns (Command API, time travel, parallel nodes, async, error handling, modern `create_agent`) | [pro_topics/README.md](pro_topics/README.md) |
| `examples/` | Part 3 — 4 real-life mini-projects (support bot, travel planner, summarizer, expense tracker) | [examples/README.md](examples/README.md) |
| `chatbot.py` + `modules/` | Part 4 — the main CLI chatbot app (RAG + tools + MCP + Mongo + Redis) | below |
| `reference/` | The full API reference — every core function's parameters & returns, in Google-style docstrings | [reference/langgraph_api_reference.py](reference/langgraph_api_reference.py) |
| `pdfs/`, `chroma_db/` | Your documents and their embedded index (gitignored) | — |
| `datastore/` | The agent's file sandbox (kept in git via `.gitkeep`) | — |
| `modules/clients.py` | Shared service clients (one Redis connection reused everywhere) | — |

---

## The Big Picture

LangGraph builds LLM apps as a **graph**: boxes (nodes) connected by arrows (edges), with one shared **state** object flowing through. LangChain chains are a straight line that runs once — LangGraph allows **branching, loops, and memory**, which is what agents need.

The single most important fact: **LangGraph only moves state between nodes.** It never calls models, never runs tools, never touches your vector store — your node functions do all of that. LangGraph is the traffic system, not the cars.

Mental model: an assembly line. The state is the product, nodes are workstations, edges are the conveyor belt — and the belt can branch or send the product backwards.

---

## 1. State — the shared data

The state is a schema (here: Pydantic `BaseModel`; `TypedDict` or dataclass also work) defining the keys every node shares.

```python
class State(BaseModel):
    messages: Annotated[list, add_messages] = []
    context: list = []
```

- **Reducers = per-key merge rules.** Default rule: **last write wins** (new value replaces old). `Annotated[list, add_messages]` changes it to **append** — that is how chat history grows instead of being overwritten.
- Choose the rule per key: conversation history must *accumulate* (`add_messages`); retrieved context must be *replaced* each search (default).
- Pydantic gives defaults (keys exist from step one), validation on every node update, and attribute access inside nodes (`state.messages`).
- At runtime everything is plain data; `invoke()` returns a plain dict of final values.

## 2. Nodes — plain functions with one contract

```python
def my_node(state: State):
    ...
    return {"some_key": new_value}
```

- Input: the **whole** current state. Output: **only the keys you want to change** — LangGraph merges your dict into the shared state. Never rebuild the full state yourself.
- A node can contain anything: an LLM call, a vector store search, plain Python. Create big clients (LLM, embeddings) **outside** the graph, reuse them in nodes.
- Tool results and retry feedback can travel as ordinary messages — a node doesn't need special retry logic if the conversation itself explains what to fix.

## 3. Edges — fixed wiring, routing, and loops

```python
graph.add_edge("a", "b")                              # fixed: after a, run b
graph.add_conditional_edges("a", router_function)     # runtime decision
```

- `START` and `END` are virtual markers for the graph's entry and exit.
- **Router function contract:** takes state, returns the **name of the next node** (a string — that's why node names matter) or `END`. It's a plain function; usually just an if-statement, sometimes an LLM call with `with_structured_output`.
- **An edge pointing backwards = a loop.** Every loop MUST have a stopping condition (an attempts counter, a "good enough" check, max rewrites). Otherwise the graph runs forever — LangGraph's default **recursion limit is 25 steps** as a safety net.
- Classic patterns: **reflection** (generate → grade → retry), **router** (choose one of N paths), **agent loop** (see §6).

## 4. Compile, invoke, stream

```python
app = graph.compile()                     # or compile(checkpointer=..., interrupt_before=[...])
result = app.invoke({"messages": [...]})   # dict in, final state dict out
```

- Without a checkpointer: the input dict **is the initial state**.
- With a checkpointer: the input is **only the new message(s)** — history loads from the thread automatically.
- `app.stream(...)` gives events as they happen (see §9).

## 5. Memory — checkpointers and threads

```python
app = graph.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "chat-1"}}
app.invoke({"messages": [new_message]}, config)
```

- A **checkpointer** saves the state after **every node execution**, and loads it back at the start of every invoke on that thread.
- A **thread** (`thread_id`) = one conversation. Each thread has its own memory; different threads know nothing about each other.
- `InMemorySaver` = RAM only, dies with the process. `SqliteSaver.from_conn_string("file.db")` = survives restarts (production: Postgres).
- Inspect saved state: `app.get_state(config)` → `.values` (the messages/data), `.next` (which node is waiting).
- Checkpointing is also the machinery behind human-in-the-loop pauses.

## 6. Tools & the agent loop

```python
@tool
def my_tool(x: int) -> str:
    """Docstring = the description the model reads."""
    ...

llm_with_tools = llm.bind_tools(TOOLS)
```

- `@tool` converts a function into a tool: **type hints become the argument schema, the docstring becomes the description** — that's ALL the model sees. The code runs only in your Python.
- The model "calls" a tool by attaching `tool_calls` to its reply: a list of `{name, args, id}` dicts. Empty list = it's done, just answering.
- You must return each result as a `ToolMessage(content=..., tool_call_id=...)` — the id is the receipt linking result to request.
- **The agent loop:** `START → model → (tool_calls? → run tools → back to model) → END`. The router just checks `last_reply.tool_calls` — **the model itself decides the route**. That's the definition of an agent: the LLM controls the graph's flow.
- The model never *executes* anything — not in LangGraph, not in old LangChain (the hidden `AgentExecutor` did what your tool node does).
- Models can request **multiple tools in one reply** (loop over `tool_calls`) and can guess args before knowing results — the loop lets them see real results and self-correct.
- One new tool = write the function + add it to the tools list. Nothing else.

## 7. RAG in a graph

- **Retrieve node:** search the vector store, write docs into a `context` key (default overwrite — fresh search each turn).
- **Generate node:** system prompt carries the context + the conversation history goes to the LLM.
- Retrieval is *just a node* — any embeddings/vector store knowledge plugs in unchanged.
- **Adaptive RAG** (lesson 7): a router decides *direct answer vs retrieve*; a grader checks if docs are relevant; a rewrite node retries with a better query (loop + guard); after max retries, answer "I don't know".
- **Agentic RAG** (lesson 12): retrieval as a *tool* — the agent decides when to search. This is what most production assistants do.

## 8. Human-in-the-loop

```python
app = graph.compile(checkpointer=InMemorySaver(), interrupt_before=["call_tools"])
```

- The graph **pauses right before** that node and `invoke()` returns early (needs a checkpointer — pausing = checkpointing).
- Inspect what's pending: `app.get_state(config).next` shows the waiting node; the last AI message's `tool_calls` shows exactly what the model wants to do.
- **Approve:** `app.invoke(None, config)` — `None` means "no new input, just continue".
- **Reject:** `app.update_state(config, {"messages": [ToolMessage(content="DENIED...", tool_call_id=id)]}, as_node="call_tools")` — write state *as if* the node ran, then `invoke(None, config)` resumes past it.

## 9. Streaming

```python
for update in app.stream(input, config, stream_mode="updates"):   ...
```

| mode | what you get | use for |
|---|---|---|
| `"updates"` | `{node_name: changed_keys}` per step | "calling tool..." status lines |
| `"values"` | full state after each step | debugging step by step |
| `"messages"` | `(token_chunk, metadata)` from LLM calls | live typewriter in a chat UI |

## 10. Subgraphs & multi-agent

- A **compiled graph can be added as a node** in another graph: `main.add_node("bot", compiled_mini_graph)`. Requirement: shared state schema (same keys).
- **Supervisor pattern:** a router node sends the question to one of N specialist subgraphs (each can be a full agent/RAG pipeline). This is the seed of multi-agent systems.

## 11. Prebuilts (nothing is magic — you built all of it by hand)

- `ToolNode(TOOLS)` — replaces a hand-written tool-dispatch node: finds functions by name, runs them, returns ToolMessages; catches tool errors and returns them as results.
- `create_react_agent(model, tools, prompt=..., checkpointer=...)` — the whole agent (model node + ToolNode + router) in one call.

---

## Function Cheat Sheet

> Full parameter lists and return structures for everything below live in [reference/langgraph_api_reference.py](reference/langgraph_api_reference.py) — one file, Google-style docstrings, cross-linked to the lessons that use each feature.

| What | Import / call | Remember |
|---|---|---|
| Build graph | `from langgraph.graph import StateGraph, START, END` | register nodes, then wire edges |
| Message reducer | `from langgraph.graph.message import add_messages` | `Annotated[list, add_messages]` |
| Run | `app = graph.compile(); app.invoke(input, config)` | returns final state dict |
| Router | `graph.add_conditional_edges(node, fn)` | fn returns node name or `END` |
| Memory (RAM) | `from langgraph.checkpoint.memory import InMemorySaver` | gone when process ends |
| Memory (file) | `from langgraph.checkpoint.sqlite import SqliteSaver` | `from_conn_string("chat.db")` |
| Threads | `{"configurable": {"thread_id": "..."}}` | one id = one conversation |
| Inspect | `app.get_state(config)` | `.values`, `.next` |
| Pause | `compile(..., interrupt_before=["node"])` | resume: `app.invoke(None, config)` |
| Edit state | `app.update_state(config, updates, as_node="node")` | the rejection path |
| Stream | `app.stream(input, config, stream_mode="...")` | updates / values / messages |
| Tools | `from langchain_core.tools import tool` | hints + docstring = schema |
| Bind tools | `llm.bind_tools(TOOLS)` | model only *requests* calls |
| Tool results | `from langchain_core.messages import ToolMessage` | must carry `tool_call_id` |
| Tool node | `from langgraph.prebuilt import ToolNode` | automatic executor |
| Whole agent | `from langgraph.prebuilt import create_react_agent` | model + tools + router |
| LLM decisions | `llm.with_structured_output(SomeClass)` | typed router/grader answers |

## Golden Rules

1. Nodes return **only the changed keys** — LangGraph merges them into state.
2. Merge rules are **per key**: default overwrite, `add_messages` appends. Pick per purpose.
3. Routers return a **node name string** or `END`, decided at runtime.
4. **Every backwards edge needs a stopping condition** (recursion limit 25 is the backstop, not the plan).
5. LangGraph never calls models or executes tools — **nodes do**. Models only *request* tool calls.
6. ToolMessage without `tool_call_id` is invalid — always copy the id.
7. With a checkpointer, **send only the new message**; history loads by `thread_id`.
8. Pausing the graph (human-in-the-loop) requires a checkpointer.
9. `invoke` output is always the **final state** — no single node ever holds all of it.
10. Simple Python wins: the complexity belongs to the graph, not the code.

---

## Lesson Index

| # | File | Concept (one line) |
|---|---|---|
| 1 | `lessons/01_first_graph.py` | State + nodes + edges; partial updates; compile & invoke |
| 2 | `lessons/02_llm_node.py` | LLM inside a node; `add_messages` reducer; message objects |
| 3 | `lessons/03_conditional_edges.py` | Routers, loops with guards, reflection/self-correction |
| 4 | `lessons/04_tool_agent.py` | Tool calling agent — manual dispatch AND prebuilt ToolNode |
| 5 | `lessons/05_memory.py` | Checkpointer, threads, multi-turn memory, `get_state` |
| 6 | `lessons/06_rag.py` | Retrieve + generate nodes; two merge rules in one state |
| 7 | `lessons/07_adaptive_rag.py` | Route direct-vs-retrieve, grade docs, rewrite-query retry loop |
| 8 | `lessons/08_human_in_loop.py` | `interrupt_before`, approve (`invoke(None)`), reject (`update_state`) |
| 9 | `lessons/09_streaming.py` | Stream modes: updates, values, messages (tokens) |
| 10 | `lessons/10_subgraphs.py` | Graphs as nodes; supervisor routing to specialists |
| 11 | `lessons/11_sqlite_memory.py` | SqliteSaver — memory that survives restarts (run twice!) |
| 12 | `lessons/12_prebuilt_agent.py` | Capstone: `create_react_agent`, RAG-as-a-tool + memory |

## The CLI Chatbot — Capstone Project

`chatbot.py` turns everything above into a real app: a CLI agent that answers from **your PDFs**, does **math**, fetches **live web data**, remembers the session, and logs every message to **MongoDB**.

**Run it:** `uv run chatbot.py`

| Piece | Where | What it does |
|---|---|---|
| PDF knowledge base | `pdfs/` | Drop PDFs here. First run embeds them all into `chroma_db/` (persistent); later runs just load the index. Delete `chroma_db/` to re-ingest. |
| Agent | `chatbot.py` | `create_agent` (modern API) + ToolNode — the model decides when to use which tool (no manual dispatch). |
| Tools | `modules/tools/` (package) | `calculator`; `get_weather`, `get_distance`, `get_currency_rate`, `search_wikipedia` (free, no-key APIs); `get_current_time`, `convert_time`, `time_between` (built on `modules/datetime_parser.py`); `fetch_url` (any web page → clean text). Split into `math_tools.py`, `web_tools.py`, `datetime_tools.py`, `url_tools.py`, `file_tools.py`, `mcp_tools.py`, `cache.py`, combined in `__init__.py`. |
| File sandbox | `modules/tools/file_tools.py` + `datastore/` | 7 sandboxed file tools (`write_file`, `read_file`, `list_directory`, `file_search`, `move_file`, `copy_file`, `file_delete`) via `FileManagementToolkit` — the agent can only touch files INSIDE `datastore/` (escape attempts are denied; paths are relative to it). |
| Redis memory | `modules/tools/redis_tools.py` + `modules/clients.py` | `redis_command` — the agent's own long-term memory. It picks any whitelisted Redis command (strings/hashes/lists/sets/ZSets, TTLs, scanning); the tool auto-prefixes every key with `REDIS_OPS:` so nothing outside the sandbox is reachable, and destructive commands (FLUSHALL, CONFIG, EVAL…) are hard-blocked. Offered only when Redis answers a ping. |
| PDF/URL search | `chatbot.py` (`search_pdfs`) | Chroma similarity search over the knowledge base — PDFs from `./pdfs` PLUS any web page you add at runtime with `/url <link>` (fetched, chunked, embedded, persisted; re-ingesting a URL replaces its chunks). `/sources` lists everything in the store. |
| MCP servers | `modules/tools/mcp_tools.py` | Connect ANY MCP server — local stdio (`uvx`/`uv`/`npx`), local HTTP, or remote with auth headers. Toggle with `MCP_ENABLED=true` in `.env` (default off). One dead server never blocks the rest; async MCP tools are auto-bridged so the sync chatbot can call them. |
| Logging | `modules/logger.py` | Production `get_logger()` — SUCCESS level, JSON or colorized output, trace/context IDs, rotation, secret redaction. Import where needed. |
| Chat log | `modules/mongo_utils.py` (`MongoUtil`) | Every message (human/ai/tool) written to MongoDB `langgraph_chatbot.messages`, message-by-message. Optional — chatbot runs fine without Mongo. |
| Cache | `modules/redis_utils.py` (`RedisStringUtil`) | Web tool results cached in Redis (weather 10 min, currency 1 h, distances/wiki 1 day). Optional — runs fine without Redis. |

**CLI commands:** `/help`, `/tools`, `/status` (session + service states), `/history`, `/clear` (forget the conversation + wipe the web cache), `/new`, `/memory` (peek at the agent's Redis memory), `/forget` (wipe that memory), `/export` (save the chat to `datastore/chat_<session>.md`), `/sources`, `/url <link>`, and `/q` / `/quit` / `/exit`. The interface is powered by **rich**: replies render as full markdown (headings, lists, code blocks), tool calls show a live spinner ("thinking… / using tools…"), and commands print styled tables.

**Env vars** (see `.env.sample`; copy to `.env`): `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDINGS_MODEL`, `MONGO_URI` (optional), `REDIS_URL` (optional), `MCP_ENABLED` (optional). Mongo, Redis, and MCP all degrade gracefully — a 2-second probe at startup, and if they're down the chatbot says so and continues.

Note on the agent API: `create_react_agent` (langgraph.prebuilt) is deprecated — it moves to `langchain.agents.create_agent` (with `system_prompt=` instead of `prompt=`) and disappears in LangGraph 2.0. `chatbot.py`, `pro_topics/` and `examples/` already use the modern `create_agent`; lessons 4 and 12 deliberately keep `create_react_agent` because that's what most tutorials still show (see `pro_topics/06` for the migration).

## Setup & Conventions

- Repo layout: `lessons/` (0-to-hero course), `pro_topics/` (production patterns), `examples/` (domain projects), `modules/` (chatbot utilities), `pdfs/` (your documents), `chroma_db/` (embedded index), `chatbot.py` (main app)
- Run everything in **WSL Ubuntu**: lessons `uv run lessons/<file>`, pro `uv run pro_topics/<file>`, examples `uv run examples/<file>`, chatbot `uv run chatbot.py`
- `.env` (gitignored, template in `.env.sample`): `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDINGS_MODEL`, `MONGO_URI`, `REDIS_URL` — always read via `os.getenv`, never hardcode
- Vector store: local Chroma (in-memory in lessons; `persist_directory=` to keep on disk)
- `chat_memory.db` (lesson 11), `chroma_db/`, and `pdfs/` are gitignored
- State schemas use **Pydantic** in this repo; keep node prints labeled (`node_name:`) so traces read like logs

## Contributing & License

Contributions are welcome — fixes, clearer explanations, new lessons or examples. Open an issue or PR.

This course is released under **CC BY-NC 4.0**: anyone may read, run, modify, share, and build on it for free, **but selling it — or teaching it for profit or benefit (paid courses, bootcamps, trainings) — is forbidden**. Full terms in [LICENSE](LICENSE). © 2026 Paras Dhiman.
