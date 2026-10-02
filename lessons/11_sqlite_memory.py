"""
Lesson 11: Real persistence - SqliteSaver

InMemorySaver (lesson 5) forgets everything when the process exits.
SqliteSaver writes checkpoints to a FILE - run this script twice and
the second run remembers the first. That is the difference between a
demo and an app: memory that survives restarts.

RUN THIS FILE TWICE:
  first run  -> the bot has never met you; you tell it your favorite planet
  second run -> the bot remembers it (from the file, not from RAM!)

A chat_memory.db file appears in the repo (already gitignored).
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()


class State(BaseModel):
    """Chat state persisted to the sqlite file by the checkpointer.

    Attributes:
        messages: conversation messages (append via add_messages).
    """

    messages: Annotated[list, add_messages] = []


llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def chat(state: State):
    """Replies with the LLM; the checkpointer then writes state to disk.

    Args:
        state: current graph state, loaded from the sqlite file.

    Returns:
        Partial update appending the AI reply.
    """
    reply = llm.invoke(state.messages)
    return {"messages": [reply]}


graph = StateGraph(State)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# from_conn_string gives a checkpointer backed by a sqlite file.
# The 'with' block opens and closes the database connection cleanly.
with SqliteSaver.from_conn_string("chat_memory.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "paras"}}

    # Is there already a saved conversation for this thread?
    saved = app.get_state(config)
    if saved.values and saved.values["messages"]:
        print("second run: found " + str(len(saved.values["messages"])) + " saved messages in the file!")
        result = app.invoke(
            {"messages": [HumanMessage(content="Do you remember my favorite planet?")]},
            config,
        )
    else:
        print("first run: no saved history. Let's teach the bot something.")
        result = app.invoke(
            {"messages": [HumanMessage(content="Hi! I'm Paras and my favorite planet is Saturn. Remember this for next time.")]},
            config,
        )

    print("bot: " + result["messages"][-1].content)

# Run me again: uv run 11_sqlite_memory.py
