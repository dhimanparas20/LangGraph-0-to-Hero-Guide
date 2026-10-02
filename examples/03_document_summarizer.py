"""
Example Project 3: Document Summarizer (map-reduce over your PDFs)

Domain: productivity - summarize long PDFs into one paragraph, store results.

What it shows:
  - The map-reduce pattern as a graph: split -> summarize each chunk (map)
    -> combine into a final summary (reduce)
  - Embedding summaries into Chroma so they are searchable afterwards
  - Storing summaries in MongoDB (optional - works without)

Run:  uv run examples/03_document_summarizer.py
Deps: OpenAI key only + at least one PDF in ./pdfs (Mongo optional).
"""

import os
import sys
from pathlib import Path
from typing import Annotated

# make `modules` importable when run as: uv run examples/03_document_summarizer.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import operator
from dotenv import load_dotenv
from pypdf import PdfReader
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

ROOT = Path(__file__).parent.parent
PDFS_DIR = ROOT / "pdfs"
MAX_CHUNKS = 3  # keep the demo cheap - raise for real use

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


class State(BaseModel):
    """Summarizer state.

    Attributes:
        chunks: text chunks of the document (map stage input).
        chunk_summaries: one summary per chunk (map stage output);
            operator.add concatenates so each map write appends.
        final_summary: the combined summary (reduce stage output).
    """

    chunks: list = []
    chunk_summaries: Annotated[list, operator.add] = []
    final_summary: str = ""


def load_chunks() -> list:
    """Extracts and splits text from the first PDF in ./pdfs.

    Returns:
        List of chunk strings (empty when no PDFs are found).
    """
    pdf_files = sorted(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("no PDFs found in", PDFS_DIR)
        return []

    reader = PdfReader(str(pdf_files[0]))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)[:MAX_CHUNKS]
    print("loaded", pdf_files[0].name, "->", len(chunks), "chunks")
    return chunks


def map_summarize(state: State) -> dict:
    """Map stage: summarizes every chunk into one bullet (plain loop).

    Args:
        state: current state holding the chunks.

    Returns:
        Partial update appending all chunk summaries at once.
    """
    summaries = []
    for i, chunk in enumerate(state.chunks):
        reply = llm.invoke([
            SystemMessage(content="Summarize this text fragment in ONE short bullet."),
            HumanMessage(content=chunk),
        ])
        summaries.append("- " + reply.content.strip())
        print("  map: summarized chunk", i + 1, "of", len(state.chunks))
    return {"chunk_summaries": summaries}


def reduce_combine(state: State) -> dict:
    """Reduce stage: fuses chunk bullets into one flowing summary.

    Args:
        state: current state holding all chunk summaries.

    Returns:
        Partial update with the final summary.
    """
    reply = llm.invoke([
        SystemMessage(content="Combine these bullets into ONE short paragraph summary. Bullets:\n" + "\n".join(state.chunk_summaries)),
        HumanMessage(content="Write the final summary."),
    ])
    return {"final_summary": reply.content.strip()}


def build_graph():
    """Builds the map-reduce summarizer graph.

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(State)
    graph.add_node("map_summarize", map_summarize)
    graph.add_node("reduce_combine", reduce_combine)
    graph.add_edge(START, "map_summarize")
    graph.add_edge("map_summarize", "reduce_combine")
    graph.add_edge("reduce_combine", END)
    return graph.compile()


def connect_mongo():
    """Connects to MongoDB for summary storage if available.

    Returns:
        A MongoUtil bound to the summaries collection, or None when offline.
    """
    try:
        from modules.mongo_utils import MongoUtil

        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017").split("?")[0]
        uri += "?serverSelectionTimeoutMS=2000&connectTimeoutMS=2000"
        util = MongoUtil(database="examples", collection="doc_summaries", connection_string=uri)
        util.count_all()
        return util
    except Exception:
        return None


def main() -> None:
    """Runs the summarize -> embed -> store pipeline for one PDF."""
    chunks = load_chunks()
    if not chunks:
        return

    app = build_graph()
    result = app.invoke({"chunks": chunks})
    summary = result["final_summary"]

    print()
    print("=== final summary ===")
    print(summary)

    # make the summary searchable too (embeddings at work)
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))
    store = Chroma.from_documents(
        [Document(page_content=summary, metadata={"kind": "summary"})],
        embeddings,
    )
    hit = store.similarity_search("what is this document about?", k=1)
    print()
    print("summary embedded and searchable - top hit starts with:", hit[0].page_content[:60], "...")

    mongo = connect_mongo()
    if mongo is not None:
        doc_id = mongo.create({"summary": summary, "chunks": len(chunks)})
        print("summary stored in mongo (id", str(doc_id)[-6:] + ")")
    else:
        print("(mongo offline - summary not persisted)")


if __name__ == "__main__":
    main()
