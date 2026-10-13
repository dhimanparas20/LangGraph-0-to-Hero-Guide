"""
Lesson 12: create_react_agent - the capstone

Everything you built by hand across lessons 4-8, prebuilt in ONE call:
    agent = create_react_agent(model, tools, prompt=..., checkpointer=...)
Inside it is exactly what you already know: a model node, a ToolNode,
and the tool-vs-done router (lesson 4, part 2).

The capstone twist: RAG AS A TOOL. Instead of a fixed retrieve -> generate
pipeline (lesson 6), the agent DECIDES when to search the lesson notes and
when to just compute or chat. That is "agentic RAG", and it is the pattern
most production assistants actually use.

Combine with the checkpointer (lesson 5) and the agent remembers turns.
"""

import os

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

from modules.llm import get_chat_model, get_embeddings

# ---------- knowledge base (same as lesson 6/7) ----------
LESSON_DOCS = [
    "Lesson 1 taught the StateGraph core: a State (shared data as a dict), nodes (plain functions that read the state and return only the keys they change), and edges (the wiring that fixes the order). START and END mark the entry and exit, compile() makes the blueprint runnable, and invoke() runs it once.",
    "Lesson 2 put an LLM inside a node and introduced the messages key with the add_messages reducer: when nodes return messages they are APPENDED instead of overwriting, which is how conversation history accumulates in LangGraph.",
    "Lesson 3 covered conditional edges: a router function reads the state and returns the name of the next node (or END). Routers enable branching and loops - edges can point backwards - and every loop needs a stopping condition such as an attempts counter.",
    "Lesson 4 built a tool-calling agent: functions decorated with @tool, llm.bind_tools(tools), the model requesting tools through reply.tool_calls, ToolMessage results paired by tool_call_id, and the agent loop model -> tools -> model until the model gives a final answer. The prebuilt ToolNode executes tool calls automatically.",
    "Lesson 5 added memory with a checkpointer: compile(checkpointer=InMemorySaver()) saves state after every step, each thread_id is its own conversation, and invoke only needs the new message because earlier history is loaded from the checkpoint.",
    "Repo conventions: state schemas use Pydantic BaseModel, code runs in WSL Ubuntu with uv, model names come from the .env file (OPENAI_MODEL and OPENAI_EMBEDDINGS_MODEL), and lessons are numbered python files at the repo root.",
]

embeddings = get_embeddings()
vectorstore = Chroma.from_texts(texts=LESSON_DOCS, embedding=embeddings)

# ---------- tools: retrieval + calculation ----------
@tool
def search_lessons(query: str) -> str:
    """Search the personal lesson notes about the LangGraph learning repo.
    Use this for ANY question about lessons, the repo, or how things work there.

    Args:
        query: the search phrase for the lesson notes.

    Returns:
        The most relevant lesson-note excerpts, joined as one string.
    """
    docs = vectorstore.similarity_search(query, k=2)
    print("   [tool] search_lessons: " + query[:50])
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers and returns the sum.

    Args:
        a: first number.
        b: second number.

    Returns:
        The sum of the two numbers.
    """
    print("   [tool] add_numbers: " + str(a) + " + " + str(b))
    return a + b

llm = get_chat_model()

# ---------- THE ONE CALL ----------
# Same building blocks you know: model with tools, memory, system prompt.
agent = create_react_agent(
    llm,
    tools=[search_lessons, add_numbers],
    prompt=(
        "You are a study assistant for Paras's LangGraph learning repo. "
        "Use search_lessons for anything about the lessons or the repo. "
        "Use add_numbers for any addition. Be concise."
    ),
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "capstone"}}

# ---------- demo 1: the agent picks the retrieval tool ----------
print("=== Q1: needs search_lessons ===")
result = agent.invoke(
    {"messages": [HumanMessage(content="According to my lesson notes, how did lesson 5 make the bot remember things?")]},
    config,
)
print("A: " + result["messages"][-1].content)

# ---------- demo 2: one question that needs BOTH tools ----------
print()
print("=== Q2: needs search_lessons AND add_numbers ===")
result = agent.invoke(
    {"messages": [HumanMessage(content="How many lessons total are described in my notes? Add 100 to that number.")]},
    config,
)
print("A: " + result["messages"][-1].content)

# ---------- demo 3: memory still works ----------
print()
print("=== Q3: memory (same thread, only the new message sent) ===")
result = agent.invoke(
    {"messages": [HumanMessage(content="Repeat the exact final number you gave me a moment ago.")]},
    config,
)
print("A: " + result["messages"][-1].content)
