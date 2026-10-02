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
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from modules import tools as tool_box
from modules.mongo_utils import MongoUtil

load_dotenv()

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
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

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
    if VECTORSTORE is not None:
        result.append(search_pdfs)
    return result


def build_agent():
    """Builds the chat agent with automatic tool dispatch and session memory.

    Returns:
        A compiled create_agent runnable (model node + ToolNode + router).
    """
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

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
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))
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
# 5. CLI LOOP
# =====================================================================

HELP_TEXT = """
commands:
  /help     show this help
  /tools    list available tools
  /history  show this session's messages
  /sources  list everything in the knowledge base (PDFs + ingested URLs)
  /url <link>  fetch a web page and embed it into the knowledge base
  /new      start a fresh session (clears the chat)
  /q        quit (also /quit or /exit)
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
    print("  bot is thinking...")

    # stream updates so the user sees which node is running (lesson 9)
    for event in agent.stream({"messages": [HumanMessage(content=text)]}, config, stream_mode="updates"):
        for node_name in event.keys():
            print("   [" + node_name + "]")

    final = agent.get_state(config).values["messages"][-1]
    logger.log(session_id, "ai", final.content)
    print("  bot: " + final.content)


def main() -> None:
    """Chatbot entry point: setup, then the read-eval-print loop.

    Initializes the vector store, Mongo logger, Redis cache and agent,
    then loops user input until an exit command is given.
    """
    global VECTORSTORE
    print("=" * 60)
    print("  LangGraph CLI Chatbot  (type /help for commands, /q to quit)")
    print("=" * 60)

    VECTORSTORE = build_vectorstore()
    logger = ChatLogger()
    if logger.enabled:
        print("[setup] mongo: chat will be logged to langgraph_chatbot.messages")
    init_cache()

    FILE_TOOLS.extend(tool_box.load_file_management_tools())

    if tool_box.mcp_enabled():
        print("[setup] mcp: MCP_ENABLED is on - connecting to MCP servers...")
        MCP_EXTRA_TOOLS.extend(tool_box.load_mcp_tools())
        print("[setup] mcp: " + str(len(MCP_EXTRA_TOOLS)) + " MCP tools available")
    else:
        print("[setup] mcp: off (set MCP_ENABLED=true to connect MCP servers)")

    agent = build_agent()

    session_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": session_id}}
    print("session: " + session_id)
    print()

    while True:
        try:
            text = input("you > ").strip()
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
            for t in tool_box_list():
                print("  - " + t.name + ": " + t.description.strip().split("\n")[0])
        elif text == "/history":
            print_history(agent, config)
        elif text == "/sources":
            print_sources()
        elif text.startswith("/url "):
            ingest_url(text[5:].strip())
        elif text == "/new":
            session_id = uuid.uuid4().hex[:8]
            config = {"configurable": {"thread_id": session_id}}
            print("  new session: " + session_id)
        else:
            try:
                chat_turn(agent, config, text, logger, session_id)
            except Exception as error:
                print("  error: " + str(error))

    print("bye! session " + session_id + " logged " + ("to mongo" if logger.enabled else "nowhere (mongo off)"))


if __name__ == "__main__":
    main()
