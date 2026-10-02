"""
Lesson 9: Streaming - watching the graph work in real time

invoke() waits silently until the whole graph finishes. stream() hands
you events AS THEY HAPPEN. Three modes cover most needs:

  stream_mode="values"   -> the FULL state after every step
  stream_mode="updates"  -> only what each step changed (tells you WHICH node ran)
  stream_mode="messages" -> LLM tokens as they are generated (the typewriter!)

This is what you plug into a UI: updates = "status: calling tool...",
messages = tokens appearing live in the chat window.
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


class State(BaseModel):
    """Chat state streamed out after every step.

    Attributes:
        messages: conversation messages (append via add_messages).
    """

    messages: Annotated[list, add_messages] = []


llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def chat(state: State):
    """Replies with the LLM; each reply becomes a stream event.

    Args:
        state: current graph state with the conversation.

    Returns:
        Partial update appending the AI reply.
    """
    reply = llm.invoke(state.messages)
    return {"messages": [reply]}


graph = StateGraph(State)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile()


# ---------- 1. stream_mode="updates": which node finished, what did it change ----------
print('=== stream_mode="updates" ===')
config1 = {"configurable": {"thread_id": "stream-updates"}}
for update in app.stream(
    {"messages": [HumanMessage(content="In one sentence: what is LangGraph?")]},
    config1,
    stream_mode="updates",
):
    # each update is a dict: {node_name: the keys that node returned}
    for node_name, changes in update.items():
        print("node finished: " + node_name + " (changed keys: " + str(list(changes.keys())) + ")")

# ---------- 2. stream_mode="values": full state after each step ----------
print()
print('=== stream_mode="values" ===')
config2 = {"configurable": {"thread_id": "stream-values"}}
for snapshot in app.stream(
    {"messages": [HumanMessage(content="Name the first 3 planets. Very short.")]},
    config2,
    stream_mode="values",
):
    # snapshot is the whole state at that moment (after input merge, after chat node)
    print("state now has " + str(len(snapshot["messages"])) + " messages")

# ---------- 3. stream_mode="messages": tokens as they are generated ----------
print()
print('=== stream_mode="messages" (typewriter) ===')
config3 = {"configurable": {"thread_id": "stream-messages"}}
for chunk, metadata in app.stream(
    {"messages": [HumanMessage(content="Count from 1 to 10, comma separated, nothing else.")]},
    config3,
    stream_mode="messages",
):
    # chunk is a piece of the AI reply (an AIMessageChunk);
    # metadata tells you which node produced it
    node = metadata.get("langgraph_node", "?")
    print("[token from " + node + "] " + repr(chunk.content))

# In a real UI you would print(chunk.content, end="") so tokens appear
# on one line as they arrive - try it!
