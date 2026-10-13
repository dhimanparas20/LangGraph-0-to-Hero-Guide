"""
Lesson 6: RAG - a retrieval node inside a LangGraph app

Nothing new about embeddings or vector stores today - you already know those.
The new part is WHERE retrieval sits in a LangGraph app:
  1. A RETRIEVE node searches the vector store and writes the found
     docs into state (the "context" key)
  2. A GENERATE node answers using that context + the conversation
  3. The checkpointer from lesson 5 keeps it working across turns

New state idea: a second state key with the DEFAULT merge rule.
"messages" keeps growing (add_messages), while "context" is fully
REPLACED on every retrieval - one state, two merge rules, each fit
for its job.

The knowledge base: summaries of lessons 1-5 of this repo. The model
has never seen them, so correct answers prove retrieval works.
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

from modules.llm import get_chat_model, get_embeddings

# ---------- 1. THE KNOWLEDGE BASE ----------
LESSON_DOCS = [
    "Lesson 1 taught the StateGraph core: a State (shared data as a dict), nodes (plain functions that read the state and return only the keys they change), and edges (the wiring that fixes the order). START and END mark the entry and exit, compile() makes the blueprint runnable, and invoke() runs it once.",
    "Lesson 2 put an LLM inside a node and introduced the messages key with the add_messages reducer: when nodes return messages they are APPENDED instead of overwriting, which is how conversation history accumulates in LangGraph.",
    "Lesson 3 covered conditional edges: a router function reads the state and returns the name of the next node (or END). Routers enable branching and loops - edges can point backwards - and every loop needs a stopping condition such as an attempts counter.",
    "Lesson 4 built a tool-calling agent: functions decorated with @tool, llm.bind_tools(tools), the model requesting tools through reply.tool_calls, ToolMessage results paired by tool_call_id, and the agent loop model -> tools -> model until the model gives a final answer. The prebuilt ToolNode executes tool calls automatically.",
    "Lesson 5 added memory with a checkpointer: compile(checkpointer=InMemorySaver()) saves state after every step, each thread_id is its own conversation, and invoke only needs the new message because earlier history is loaded from the checkpoint.",
    "Repo conventions: state schemas use Pydantic BaseModel, code runs in WSL Ubuntu with uv, model names come from the .env file (OPENAI_MODEL and OPENAI_EMBEDDINGS_MODEL), and lessons are numbered python files at the repo root.",
]

embeddings = get_embeddings()

# In-memory Chroma collection built from the docs above.
# (A real app would pass persist_directory="chroma_db" to keep it on disk.)
vectorstore = Chroma.from_texts(texts=LESSON_DOCS, embedding=embeddings)

# ---------- 2. STATE ----------
class State(BaseModel):
    """RAG state with two keys, two merge rules.

    Attributes:
        messages: conversation (appends via add_messages).
        context: retrieved doc texts; plain overwrite so every new
            search replaces the previous context.
    """

    messages: Annotated[list, add_messages] = []   # appends: conversation grows
    context: list = []                             # overwrites: latest retrieval wins

llm = get_chat_model()

# ---------- 3. RETRIEVE NODE ----------
# Just a node like any other - it happens to talk to a vector store.
def retrieve(state: State):
    """Searches the vector store with the newest human message.

    Args:
        state: current graph state; the last message is the question.

    Returns:
        Partial update replacing the context with the top doc texts.
    """
    question = state.messages[-1].content          # the newest human message
    docs = vectorstore.similarity_search(question, k=2)
    texts = [doc.page_content for doc in docs]

    print("retrieve: top " + str(len(texts)) + " docs")
    for text in texts:
        print("   - " + text[:55] + "...")

    return {"context": texts}                      # replaces any older context

# ---------- 4. GENERATE NODE ----------
def generate(state: State):
    """Answers using the retrieved context plus the conversation history.

    Args:
        state: current graph state with question and retrieved context.

    Returns:
        Partial update appending the grounded AI answer.
    """
    context_text = "\n\n".join(state.context)

    system_prompt = (
        "You answer questions about a LangGraph learning repo. "
        "Use ONLY the context below. If the answer is not in the context, say you don't know.\n\n"
        "Context:\n" + context_text
    )

    # The system prompt carries the retrieved context; the conversation history
    # (loaded from the checkpoint on later turns) keeps the dialogue coherent.
    reply = llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
    print("generate: answered")
    return {"messages": [reply]}

# ---------- 5. THE GRAPH ----------
graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile(checkpointer=InMemorySaver())   # lesson 5: memory on

config = {"configurable": {"thread_id": "rag-chat"}}

# ---------- 6. TWO TURNS ----------
# Turn 1: a question only answerable from the lesson 4 doc.
result = app.invoke(
    {"messages": [HumanMessage(content="How does the agent loop know when to stop calling tools?")]},
    config,
)
print()
print("turn 1 answer: " + result["messages"][-1].content)
print()

# Turn 2 (same thread): only the new message is sent; history comes from
# the checkpoint, and retrieval runs fresh for this question too.
result = app.invoke(
    {"messages": [HumanMessage(content="And how did we make the chatbot remember earlier turns?")]},
    config,
)
print()
print("turn 2 answer: " + result["messages"][-1].content)
