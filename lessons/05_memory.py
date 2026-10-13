"""
Lesson 5: Memory - checkpointing and threads

New ideas today:
  1. A CHECKPOINTER saves the state after every step (node execution)
  2. A THREAD (thread_id) is one conversation - each thread keeps
     its own saved state
  3. invoke() now works like a chat: you send ONLY the new message,
     the old ones are loaded from the checkpoint automatically

The test: tell the bot your name and favorite planet, then in a SEPARATE
invoke ask it to recall them. Also ask the same question in a DIFFERENT
thread - it should know nothing there.
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

from modules.llm import get_chat_model

class State(BaseModel):
    """Chat state kept alive between invokes by the checkpointer.

    Attributes:
        messages: conversation messages (append via add_messages).
    """

    messages: Annotated[list, add_messages] = []

llm = get_chat_model()

def chat(state: State):
    """Replies to the conversation with the LLM.

    Args:
        state: current graph state (loaded from the thread checkpoint).

    Returns:
        Partial update appending the AI reply.
    """
    reply = llm.invoke(state.messages)
    return {"messages": [reply]}

graph = StateGraph(State)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# THE NEW PART: compile with a checkpointer.
# Now LangGraph saves the state after every step, and loads it back
# at the start of every invoke on the same thread.
# InMemorySaver keeps checkpoints in RAM (gone when the script ends);
# real apps use a database-backed saver (sqlite, postgres) - same idea.
app = graph.compile(checkpointer=InMemorySaver())

# Each thread_id is one conversation with its own memory.
config_thread_1 = {"configurable": {"thread_id": "chat-with-paras"}}

# ---- turn 1: tell it something ----
result = app.invoke(
    {"messages": [HumanMessage(content="Hi! I'm Paras and my favorite planet is Saturn.")]},
    config_thread_1,
)
print("turn 1 reply: " + result["messages"][-1].content)
print()

# ---- turn 2: SAME thread. Note the input contains ONLY the new message -
# the earlier messages come from the checkpoint, not from us. ----
result = app.invoke(
    {"messages": [HumanMessage(content="What is my name and my favorite planet?")]},
    config_thread_1,
)
print("turn 2 reply: " + result["messages"][-1].content)
print()

# ---- same question, DIFFERENT thread: this one has no memory of us ----
config_thread_2 = {"configurable": {"thread_id": "chat-with-a-stranger"}}
result = app.invoke(
    {"messages": [HumanMessage(content="What is my name and my favorite planet?")]},
    config_thread_2,
)
print("other thread reply: " + result["messages"][-1].content)
print()

# ---- peek at what is actually saved for thread 1 ----
saved = app.get_state(config_thread_1)
print("=== saved conversation of thread 1 (" + str(len(saved.values["messages"])) + " messages) ===")
for message in saved.values["messages"]:
    print(message.type + ": " + message.content)
