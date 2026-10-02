"""
Pro Lesson 4: Async graphs - ainvoke and async nodes

Everything so far was sync: invoke() blocks until done. Real services are
async (web servers, many API calls at once). LangGraph supports this
natively: define async nodes (async def + await) and run the graph with
await app.ainvoke(...) or async for ... in app.astream(...).

Why it matters: while one node awaits an LLM/API response, the event loop
can run other work - that is how you serve many users from one process.
"""

import asyncio
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
    """State for a tiny two-step async pipeline.

    Attributes:
        messages: conversation messages (append via add_messages).
        shouty: the answer converted to uppercase by the second node.
    """

    messages: Annotated[list, add_messages] = []
    shouty: str = ""


llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


async def ask_llm(state: State) -> dict:
    """Async node: awaits the model without blocking the event loop.

    Args:
        state: current graph state with the user's question.

    Returns:
        Partial update appending the model's reply.
    """
    print("ask_llm: awaiting the model...")
    reply = await llm.ainvoke(state.messages)
    print("ask_llm: got reply")
    return {"messages": [reply]}


async def make_shouty(state: State) -> dict:
    """Async node: uppercases the latest reply (trivially async to show the pattern).

    Args:
        state: current graph state containing the reply.

    Returns:
        Partial update with the uppercased answer.
    """
    await asyncio.sleep(0.1)  # stand-in for any awaited I/O
    return {"shouty": state.messages[-1].content.upper()}


def build_graph():
    """Builds and compiles the async graph (same API as sync graphs).

    Returns:
        A compiled LangGraph runnable supporting ainvoke/astream.
    """
    graph = StateGraph(State)
    graph.add_node("ask_llm", ask_llm)
    graph.add_node("make_shouty", make_shouty)
    graph.add_edge(START, "ask_llm")
    graph.add_edge("ask_llm", "make_shouty")
    graph.add_edge("make_shouty", END)
    return graph.compile()


async def run_demo() -> None:
    """Runs the graph twice: once with ainvoke, once token-streaming with astream.

    Shows both async entry points and that sync-style config still applies.
    """
    app = build_graph()

    print("=== await app.ainvoke ===")
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="Say exactly: async is easy")]}
    )
    print("shouty:", result["shouty"])

    print()
    print("=== async for ... app.astream(stream_mode='updates') ===")
    config = {"configurable": {"thread_id": "async-demo"}}
    async for event in app.astream(
        {"messages": [HumanMessage(content="Say exactly: streams too")]},
        config,
        stream_mode="updates",
    ):
        for node_name in event:
            print("  node finished:", node_name)


if __name__ == "__main__":
    asyncio.run(run_demo())
