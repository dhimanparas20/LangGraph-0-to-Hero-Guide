"""
Pro Lesson 5: Error handling - flaky tools, retries, and recursion limits

Production agents fail in three ways, and there is an answer for each:

  1. A TOOL fails (API down, bad input).
     MIGRATION GOTCHA: the old create_react_agent auto-caught tool
     exceptions (handle_tool_errors=True) and fed the error text back to
     the model. The modern create_agent does NOT - exceptions crash the
     graph unless YOU decide the policy. The most common pattern (used
     here): the tool catches its own error and RETURNS it as text, so the
     model reads the error and retries. Works identically everywhere.
  2. The LLM call itself fails (rate limit, timeout).
     Wrap the model with .with_retry(...) - a LangChain runnable feature.
     NOTE: wrap it for DIRECT calls only; an agent needs the raw model
     because it calls .bind_tools() internally (a retried wrapper cannot).
  3. The agent loops forever (model keeps calling tools).
     invoke(config={"recursion_limit": N}) aborts with GraphRecursionError.

This lesson demonstrates all three, with a deliberately flaky tool.
Uses the modern create_agent API (see pro lesson 06).
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

load_dotenv()

# module-level counter: the tool FAILS the first 2 times, then works.
# (A state-free trick so the demo is deterministic.)
CALL_COUNT = {"n": 0}


@tool
def fetch_price(product: str) -> str:
    """Fetches the price of a product from the shop API. Call it for any price question.

    Args:
        product: name of the product to look up.

    Returns:
        The price string, or an ERROR:... message the agent can retry on.
    """
    CALL_COUNT["n"] += 1
    if CALL_COUNT["n"] <= 2:
        # error-as-text instead of raising: the model reads this and retries
        return "ERROR: shop API timeout (attempt " + str(CALL_COUNT["n"]) + "), try again"
    return product + " costs 499 rupees"


def build_agent():
    """Builds the agent: model + tools over the flaky price tool.

    Returns:
        A compiled agent runnable.
    """
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return create_agent(
        llm,
        tools=[fetch_price],
        system_prompt="Use fetch_price for any price question. If a tool returns an ERROR message, retry it.",
    )


def demo_retry_wrapper() -> None:
    """Shows the .with_retry pattern for direct LLM calls (failure type 2)."""
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    retried_llm = llm.with_retry(stop_after_attempt=3)
    reply = retried_llm.invoke([HumanMessage(content="Say exactly: ok")])
    print("retry-wrapped direct call succeeded:", reply.content)


def demo_flaky_tool(agent) -> None:
    """Runs the agent whose tool fails twice; error-as-text lets the model retry.

    Args:
        agent: compiled agent from build_agent().
    """
    config = {"configurable": {"thread_id": "errors"}, "recursion_limit": 10}
    print("=== demo 1: tool fails twice, agent reads errors and retries ===")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the price of a ZetaPhone?")]},
        config,
    )
    print("final:", result["messages"][-1].content)
    print("(tool was called", CALL_COUNT["n"], "times: 2 errors + 1 success)")


def demo_recursion_limit(agent) -> None:
    """Shows GraphRecursionError being caught when the step limit is too low.

    Args:
        agent: compiled agent from build_agent().
    """
    print()
    print("=== demo 2: catching GraphRecursionError ===")
    try:
        agent.invoke(
            {"messages": [HumanMessage(content="Price of a ZetaPhone?")]},
            {"configurable": {"thread_id": "loop"}, "recursion_limit": 2},
        )
    except GraphRecursionError:
        print("caught GraphRecursionError: the graph hit the step limit and stopped safely")


def main() -> None:
    """Runs all three error-handling demos in order."""
    demo_retry_wrapper()
    print()
    agent = build_agent()
    demo_flaky_tool(agent)
    demo_recursion_limit(agent)


if __name__ == "__main__":
    main()
