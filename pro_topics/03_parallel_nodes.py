"""
Pro Lesson 3: Parallel nodes - fan-out, reducers, and join

Edges can fan out: if node A has edges to B, C and D, all three run in the
SAME superstep (LangGraph runs them concurrently and waits for all before
continuing). Two rules make this safe:

  1. Parallel nodes must NOT write the same plain key - default merge is
     "last write wins", so they would silently overwrite each other.
  2. Give the shared key a REDUCER that combines writes instead:
     Annotated[list, operator.add] concatenates lists.

Here: one "fan_out" node dispatches work to three analyst nodes, each
appends its result to the shared list, and "join" runs only after ALL
three finished. This is map-reduce inside a graph.
"""

import operator
from typing import Annotated

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END


class State(BaseModel):
    """State with a reducer-protected list for parallel writes.

    Attributes:
        topic: the topic every analyst works on.
        results: analyst outputs; operator.add CONCATENATES the lists each
            parallel node returns, instead of overwriting them.
    """

    topic: str = ""
    results: Annotated[list, operator.add] = []


def fan_out(state: State) -> dict:
    """Sets the shared topic all analysts will analyze.

    Args:
        state: current graph state.

    Returns:
        Partial update with the topic.
    """
    print("fan_out: topic =", "renewable energy")
    return {"topic": "renewable energy"}


def optimist(state: State) -> dict:
    """Optimist analyst appending its take to the results list.

    Args:
        state: current graph state with the topic.

    Returns:
        Partial update appending one result string.
    """
    return {"results": ["OPTIMIST: " + state.topic + " will grow fast"]}


def pessimist(state: State) -> dict:
    """Pessimist analyst appending its take to the results list.

    Args:
        state: current graph state with the topic.

    Returns:
        Partial update appending one result string.
    """
    return {"results": ["PESSIMIST: " + state.topic + " faces hard limits"]}


def realist(state: State) -> dict:
    """Realist analyst appending its take to the results list.

    Args:
        state: current graph state with the topic.

    Returns:
        Partial update appending one result string.
    """
    return {"results": ["REALIST: " + state.topic + " grows steadily with policy"]}


def join(state: State) -> dict:
    """Runs only after ALL parallel analysts finished; prints the merged list.

    Args:
        state: current graph state holding all analyst results.

    Returns:
        Empty update (this node only reports).
    """
    print("join: all analysts done, collected", len(state.results), "results")
    for line in state.results:
        print("   -", line)
    return {}


def build_graph():
    """Builds the fan-out / fan-in graph.

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(State)
    graph.add_node("fan_out", fan_out)
    graph.add_node("optimist", optimist)
    graph.add_node("pessimist", pessimist)
    graph.add_node("realist", realist)
    graph.add_node("join", join)

    graph.add_edge(START, "fan_out")
    # FAN-OUT: three edges from the same node -> they run in parallel
    graph.add_edge("fan_out", "optimist")
    graph.add_edge("fan_out", "pessimist")
    graph.add_edge("fan_out", "realist")
    # FAN-IN / JOIN: "join" waits for ALL three to finish
    graph.add_edge("optimist", "join")
    graph.add_edge("pessimist", "join")
    graph.add_edge("realist", "join")
    graph.add_edge("join", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({})
    print("final results count:", len(result["results"]))
