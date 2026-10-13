"""
CLI Chatbot - the capstone project of this repo.

Everything from the lessons combined into one app:
  - RAG: PDFs from ./pdfs are embedded into a persistent local Chroma
    store on FIRST run (chroma_db/); later runs just load it
  - Agent: create_agent + ToolNode (automatic tool dispatch - the model
    decides when to search the PDFs, do math, or call public APIs)
  - Memory: InMemorySaver checkpointer with a thread per chat session
  - Persistence: every message is logged to MongoDB (message-by-message,
    so history is safe even if the app crashes) - optional, degrades
    gracefully when Mongo is not reachable
  - Cache: web tool results cached in Redis when available - optional
  - CLI: /help /tools /history /new /q (/quit, /exit also work)

Run:  uv run chatbot.py
"""

import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# load .env BEFORE importing modules - modules/clients.py and others read
# env vars (REDIS_URL, DATASTORE_DIR, ...) at import time
load_dotenv()

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from modules import tools as tool_box
from modules.llm import get_chat_model, get_embeddings, llm_provider, resolve_chat_model_name
from modules.mongo_utils import MongoUtil

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# the one console used for all output (rich auto-detects colors/TTY)
console = Console()

# ---------- paths (pathlib everywhere) ----------
ROOT = Path(__file__).parent
PDFS_DIR = ROOT / "pdfs"
CHROMA_DIR = ROOT / "chroma_db"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "")

# filled by main() before the agent is built
VECTORSTORE = None

# MCP tools loaded at startup when MCP_ENABLED=true (see modules/tools/mcp_tools.py)
MCP_EXTRA_TOOLS = []

# sandboxed file tools for the datastore/ directory (always on)
FILE_TOOLS = []

# redis memory tool (offered when Redis answers a ping)
REDIS_TOOLS = []


# =====================================================================
# 1. VECTOR STORE: ingest PDFs on first run, load them on later runs
# =====================================================================

def build_vectorstore():
    """Returns the PDF knowledge base, ingesting ./pdfs on the first run.

    On the first run (no chroma_db/ folder) every PDF in ./pdfs is read with
    pypdf, split into chunks, embedded, and persisted to chroma_db/. Later
    runs load the existing index instead of re-embedding.

    Returns:
        A Chroma vector store, or None when no PDF text could be ingested
        (PDF search is then disabled for the session).
    """
    embeddings = get_embeddings()

    # not the first run -> the persisted index is already there
    if CHROMA_DIR.exists():
        print("[setup] loading existing Chroma index from chroma_db/")
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    pdf_files = sorted(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[setup] WARNING: no PDFs found in ./pdfs - PDF search will be unavailable.")
        print("[setup]        add PDFs and delete the chroma_db/ folder to rebuild.")
        return None

    print("[setup] first run: embedding " + str(len(pdf_files)) + " PDF(s) from ./pdfs ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            print("[setup]   " + pdf_path.name + ": no extractable text, skipped")
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata={"source": pdf_path.name}))
        print("[setup]   " + pdf_path.name + ": " + str(len(chunks)) + " chunks")

    if not all_chunks:
        return None

    vectorstore = Chroma.from_documents(all_chunks, embeddings, persist_directory=str(CHROMA_DIR))
    print("[setup] stored " + str(len(all_chunks)) + " chunks in chroma_db/")
    return vectorstore


@tool
def search_pdfs(query: str) -> str:
    """Search the user's documents (PDFs from ./pdfs and web pages added via /url).
    Use this for ANY question that might be answered by the user's own documents.

    Args:
        query: the search phrase to look for in the knowledge base.

    Returns:
        The most relevant excerpts, labeled with their source (file or URL).
    """
    docs = VECTORSTORE.similarity_search(query, k=3)
    if not docs:
        return "No PDF content available."
    parts = []
    for doc in docs:
        parts.append("From " + doc.metadata.get("source", "pdf") + ":\n" + doc.page_content)
    return "\n\n".join(parts)


# =====================================================================
# 2. MONGO CHAT LOGGING (optional - message by message)
# =====================================================================

class ChatLogger:
    """Logs every chat message to MongoDB, disabling itself when Mongo is down.

    Attributes:
        enabled: whether logging is currently active.
        util: the MongoUtil handle bound to langgraph_chatbot.messages,
            or None when Mongo was unreachable at startup.
    """

    def __init__(self) -> None:
        """Connects to MongoDB with short timeouts; no failure is fatal.

        A cheap count_all() probe verifies the server is actually reachable
        (MongoClient itself connects lazily and would not fail here).
        """
        self.enabled = False
        self.util = None
        try:
            # short timeouts so a missing Mongo doesn't hang the chatbot
            uri = MONGO_URI.split("?")[0] + "?serverSelectionTimeoutMS=2000&connectTimeoutMS=2000"
            self.util = MongoUtil(database="langgraph_chatbot", collection="messages", connection_string=uri)
            self.util.count_all()   # cheap probe - fails fast if server is unreachable
            self.enabled = True
        except Exception as error:
            print("[setup] mongo logging disabled:", str(error).split("\n")[0])

    def log(self, session_id: str, role: str, content: str) -> None:
        """Stores one message; turns itself off permanently on the first failure.

        Args:
            session_id: the chat session this message belongs to.
            role: who produced the message (human / ai / tool).
            content: the message text.
        """
        if not self.enabled:
            return
        try:
            self.util.create({
                "session_id": session_id,
                "role": role,
                "content": content,
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
        except Exception:
            self.enabled = False   # first failure turns logging off, chat continues
            print("[mongo] unreachable - chat logging turned off for this session")


# =====================================================================
# 3. REDIS CACHE (optional)
# =====================================================================

def init_cache() -> None:
    """Gives the web tools a Redis cache if REDIS_URL points to a live server.

    When Redis is unavailable (or REDIS_URL unset) the chatbot continues
    normally - tools just skip caching.
    """
    if not REDIS_URL:
        print("[setup] redis: REDIS_URL not set - caching off")
        return
    try:
        from modules.redis_utils import RedisStringUtil

        cache = RedisStringUtil(url=REDIS_URL, prefix="CHATBOT:CACHE")
        tool_box.set_cache(cache)
        print("[setup] redis: connected - web tool results will be cached")
    except Exception as error:
        print("[setup] redis: not reachable - caching off (" + str(error) + ")")


# =====================================================================
# 4. THE AGENT
# =====================================================================

def tool_box_list() -> list:
    """Collects every tool the agent may use.

    Returns:
        The built-in tools from modules/tools (math, web, datetime), any
        MCP tools loaded at startup, plus search_pdfs when a vector
        store exists.
    """
    result = [
        tool_box.calculator,
        tool_box.get_weather,
        tool_box.get_distance,
        tool_box.get_currency_rate,
        tool_box.search_wikipedia,
        tool_box.get_current_time,
        tool_box.convert_time,
        tool_box.time_between,
        tool_box.fetch_url,
    ]
    result.extend(MCP_EXTRA_TOOLS)
    result.extend(FILE_TOOLS)
    result.extend(REDIS_TOOLS)
    if VECTORSTORE is not None:
        result.append(search_pdfs)
    return result


def build_agent():
    """Builds the chat agent with automatic tool dispatch and session memory.

    Returns:
        A compiled create_agent runnable (model node + ToolNode + router).
    """
    llm = get_chat_model()

    all_tools = tool_box_list()
    if not all_tools:
        raise SystemExit("No tools available - add PDFs or check modules/tools.py")

    return create_agent(
        llm,
        tools=all_tools,
        system_prompt=(
            "You are a helpful CLI study assistant. "
            "Use search_pdfs for anything that might be in the user's PDF documents. "
            "Use the calculator for math, the time tools for any current-time, timezone "
            "conversion or duration question, the web tools for weather, distances, "
            "currency rates and Wikipedia lookups, fetch_url to read any link the user "
            "shares, the file tools to manage files in the datastore sandbox "
            "(file paths are RELATIVE to it: 'notes.txt' means datastore/notes.txt), "
            "redis_command to use your own long-term memory in Redis (remember and "
            "recall facts across sessions with SET/GET), "
            "and any web-search tools for questions needing fresh information "
            "from the internet. Be concise and friendly."
        ),
        checkpointer=InMemorySaver(),   # memory within this session (thread per session)
    )


def ingest_url(url: str) -> None:
    """Fetches a web page and embeds it into the persistent knowledge base.

    The page text is fetched (modules/tools/url_tools), split into chunks,
    and added to the chroma_db/ Chroma store with the URL as its source -
    so later questions can retrieve it via search_pdfs. Re-ingesting the
    same URL replaces its old chunks.

    Args:
        url: the page URL (http:// or https://).
    """
    global VECTORSTORE

    from modules.tools.url_tools import fetch_url_text

    try:
        text = fetch_url_text(url)
    except Exception as error:
        print("  error: " + str(error))
        return

    # create the store on demand if the first run had no PDFs at all
    if VECTORSTORE is None:
        embeddings = get_embeddings()
        VECTORSTORE = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        print("  (note: created a fresh vector store - restart to get the search_pdfs tool)")

    # replace any previous chunks from this same URL
    existing = VECTORSTORE.get(where={"source": url})
    if existing and existing.get("ids"):
        VECTORSTORE.delete(ids=existing["ids"])

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]
    VECTORSTORE.add_documents(docs)
    print("  ingested " + str(len(chunks)) + " chunks from " + url)


def print_sources() -> None:
    """Lists every source in the knowledge base with its chunk count."""
    if VECTORSTORE is None:
        print("  (knowledge base is empty)")
        return
    data = VECTORSTORE.get(include=["metadatas"])
    counts = {}
    for meta in data["metadatas"]:
        source = meta.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    print("  knowledge base (" + str(len(data["metadatas"])) + " chunks):")
    for source in sorted(counts):
        print("   - " + str(counts[source]) + " chunks | " + source)


# =====================================================================
# 5. CLI COMMANDS
# =====================================================================

def clear_session(agent, config: dict, session_id: str) -> None:
    """Forgets the session's conversation and clears the web tool cache.

    Deletes the thread's checkpoints (the conversation memory) and, when
    Redis is reachable, every CHATBOT:CACHE:* entry so web tools fetch
    fresh data again.

    Args:
        agent: the compiled agent runnable.
        config: thread config identifying the session.
        session_id: the session id, for reporting.
    """
    forgot_conversation = False
    try:
        agent.checkpointer.delete_thread(session_id)
        forgot_conversation = True
    except Exception:
        # very old savers have no delete_thread - fall back to a new session
        pass

    cache_cleared = 0
    if tool_box.redis_available():
        from modules.clients import r

        keys = r.keys("CHATBOT:CACHE:*")
        if keys:
            cache_cleared = r.delete(*keys)

    if forgot_conversation:
        console.print("[dim]conversation cleared (this session starts fresh)[/]")
    else:
        console.print("[yellow]checkpointer cannot delete - starting a NEW session instead[/]")
    console.print("[dim]web cache cleared: " + str(cache_cleared) + " entries[/]")


def show_status(agent, config: dict, session_id: str, logger: ChatLogger) -> None:
    """Prints session info and the live state of every optional service.

    Args:
        agent: the compiled agent runnable.
        config: thread config identifying the session.
        session_id: the current session id.
        logger: the Mongo ChatLogger (its enabled flag is reported).
    """
    state = agent.get_state(config)
    message_count = len(state.values.get("messages", [])) if state.values else 0

    kb_chunks = 0
    if VECTORSTORE is not None:
        kb_chunks = len(VECTORSTORE.get(include=["metadatas"])["metadatas"])

    table = Table(title="status", show_header=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("session", session_id + "  (" + str(message_count) + " messages)")
    table.add_row("provider", llm_provider())
    table.add_row("model", resolve_chat_model_name())
    table.add_row("tools", str(len(tool_box_list())))
    table.add_row("mongo logging", "ON" if logger.enabled else "off")
    table.add_row("redis memory", "ON" if REDIS_TOOLS else "off")
    table.add_row("mcp tools", str(len(MCP_EXTRA_TOOLS)) + " connected" if MCP_EXTRA_TOOLS else "off")
    table.add_row("file sandbox", str(len(FILE_TOOLS)) + " tools at " + str(tool_box.DATASTORE_DIR))
    table.add_row("knowledge base", str(kb_chunks) + " chunks")
    console.print(table)


def show_memory() -> None:
    """Lists the keys in the agent's long-term Redis memory (REDIS_OPS:*)."""
    if not tool_box.redis_available():
        console.print("[dim]redis memory: off[/]")
        return
    from modules.clients import r

    keys = sorted(r.keys("REDIS_OPS:*"))
    if not keys:
        console.print("[dim]redis memory is empty[/]")
        return
    table = Table(title="agent long-term memory (REDIS_OPS:*)")
    table.add_column("key", style="cyan")
    table.add_column("type", style="dim")
    table.add_column("ttl", justify="right")
    for key in keys:
        ttl = r.ttl(key)
        table.add_row(key, r.type(key), str(ttl) if ttl >= 0 else "forever")
    console.print(table)


def forget_memory() -> None:
    """Wipes every key in the agent's long-term Redis memory."""
    if not tool_box.redis_available():
        console.print("[dim]redis memory: off[/]")
        return
    from modules.clients import r

    keys = r.keys("REDIS_OPS:*")
    if not keys:
        console.print("[dim]redis memory is already empty[/]")
        return
    count = r.delete(*keys)
    console.print("[yellow]wiped " + str(count) + " memory keys[/]")


def export_session(agent, config: dict, session_id: str) -> None:
    """Saves this conversation to datastore/chat_<session>.md as markdown.

    Args:
        agent: the compiled agent runnable.
        config: thread config identifying the session.
        session_id: the session id (used in the filename).
    """
    state = agent.get_state(config)
    messages = state.values.get("messages", []) if state.values else []
    if not messages:
        console.print("[dim]nothing to export yet[/]")
        return

    lines = ["# chat session " + session_id, "_" + datetime.now().isoformat(timespec="seconds") + "_", ""]
    for message in messages:
        if message.type == "ai" and not message.content:
            continue
        role = "you" if message.type == "human" else ("bot" if message.type == "ai" else message.type)
        lines.append("## " + role)
        lines.append("")
        lines.append(message.content)
        lines.append("")

    path = tool_box.get_datastore_dir() / ("chat_" + session_id + ".md")
    path.write_text("\n".join(lines), encoding="utf-8")
    console.print("[dim]exported to " + str(path) + "[/]")


# =====================================================================
# 6. CLI LOOP
# =====================================================================

HELP_TEXT = """
commands:
  /help         show this help
  /tools        list available tools
  /history      show this session's messages
  /status       session info + which services are live
  /clear        forget this session's conversation + clear the web cache
  /new          start a fresh session (new id, old one kept)
  /memory       peek at the agent's long-term Redis memory (REDIS_OPS:*)
  /forget       wipe the agent's long-term Redis memory
  /export       save this conversation to datastore/chat_<session>.md
  /sources      list the knowledge base (PDFs + ingested URLs)
  /url <link>   fetch a web page and embed it into the knowledge base
  /q            quit (also /quit or /exit)
anything else is sent to the chatbot.
"""

EXIT_WORDS = {"/q", "/quit", "/exit"}


def print_history(agent, config: dict) -> None:
    """Prints the current session's messages from the checkpointed state.

    Args:
        agent: the compiled agent runnable.
        config: thread config identifying the session.
    """
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    if not messages:
        print("  (no messages yet)")
        return
    for message in messages:
        if message.type == "ai" and not message.content:
            continue   # intermediate ai message that only requested tools
        role = "you" if message.type == "human" else message.type
        print("  " + role + ": " + message.content[:200])


def chat_turn(agent, config: dict, text: str, logger: ChatLogger, session_id: str) -> None:
    """Runs one question through the agent and prints the answer.

    Streams node-level updates so the user sees progress (lesson 9), then
    reads the final answer from the checkpointed state.

    Args:
        agent: the compiled agent runnable.
        config: thread config identifying the session.
        text: the user's message.
        logger: ChatLogger used to persist both sides of the turn.
        session_id: current session id for logging.
    """
    logger.log(session_id, "human", text)

    # live spinner while the agent works; the label follows the graph's progress
    with console.status("[dim]bot is thinking...[/]", spinner="dots") as status:
        for event in agent.stream({"messages": [HumanMessage(content=text)]}, config, stream_mode="updates"):
            for node_name in event.keys():
                if "tool" in node_name:
                    status.update("[dim]bot is using tools...[/]")
                else:
                    status.update("[dim]bot is thinking...[/]")

    final = agent.get_state(config).values["messages"][-1]
    logger.log(session_id, "ai", final.content)
    console.print(Markdown(final.content))
    console.print()


def main() -> None:
    """Chatbot entry point: setup, then the read-eval-print loop.

    Initializes the vector store, Mongo logger, Redis cache and agent,
    then loops user input until an exit command is given.
    """
    global VECTORSTORE
    console.print(Panel.fit(
        "[bold cyan]LangGraph CLI Chatbot[/]\n"
        "[dim]markdown-rendered replies - type /help for commands, /q to quit[/]",
        border_style="cyan",
    ))

    VECTORSTORE = build_vectorstore()
    logger = ChatLogger()
    if logger.enabled:
        print("[setup] mongo: chat will be logged to langgraph_chatbot.messages")
    init_cache()

    FILE_TOOLS.extend(tool_box.load_file_management_tools())

    if tool_box.redis_available():
        REDIS_TOOLS.append(tool_box.redis_command)
        print("[setup] redis memory: ON (agent can use redis_command in the REDIS_OPS sandbox)")

    if tool_box.mcp_enabled():
        print("[setup] mcp: MCP_ENABLED is on - connecting to MCP servers...")
        MCP_EXTRA_TOOLS.extend(tool_box.load_mcp_tools())
        print("[setup] mcp: " + str(len(MCP_EXTRA_TOOLS)) + " MCP tools available")
    else:
        print("[setup] mcp: off (set MCP_ENABLED=true to connect MCP servers)")

    agent = build_agent()

    session_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": session_id}}
    console.print("[dim]session: " + session_id + "[/]")
    console.print()

    while True:
        try:
            text = Prompt.ask("[bold green]you >[/]").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        if text.lower() in EXIT_WORDS:
            break

        if text == "/help":
            print(HELP_TEXT)
        elif text == "/tools":
            table = Table(title="agent tools", show_header=True)
            table.add_column("tool", style="bold cyan", no_wrap=True)
            table.add_column("what it does")
            for t in tool_box_list():
                table.add_row(t.name, t.description.strip().split("\n")[0])
            console.print(table)
        elif text == "/status":
            show_status(agent, config, session_id, logger)
        elif text == "/clear":
            clear_session(agent, config, session_id)
            config = {"configurable": {"thread_id": session_id}}
        elif text == "/history":
            print_history(agent, config)
        elif text == "/memory":
            show_memory()
        elif text == "/forget":
            forget_memory()
        elif text == "/export":
            export_session(agent, config, session_id)
        elif text == "/sources":
            print_sources()
        elif text.startswith("/url "):
            ingest_url(text[5:].strip())
        elif text == "/new":
            session_id = uuid.uuid4().hex[:8]
            config = {"configurable": {"thread_id": session_id}}
            console.print("[dim]new session: " + session_id + "[/]")
        else:
            try:
                chat_turn(agent, config, text, logger, session_id)
            except Exception as error:
                console.print("[red]error: " + str(error) + "[/]")

    console.print("[dim]bye! session " + session_id + " logged "
                  + ("to mongo" if logger.enabled else "nowhere (mongo off)") + "[/]")


if __name__ == "__main__":
    main()
