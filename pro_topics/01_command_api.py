"""
Pro Lesson 1: The Command API - routing and updating in one place

The classic style (lessons 3-12): a node returns a dict of updates, and a
SEPARATE router function + add_conditional_edges decides the next node.
The Command API merges both jobs: a node can return

    Command(goto="next_node_name", update={"key": value})

meaning "apply this update AND go to that node". Same graph, less wiring -
and since the node itself decides, it can route on information it just
computed, without round-tripping it through state first.

Command works with plain edges too: goto=END finishes the graph.
"""

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class State(BaseModel):
    """Graph state holding a number and its classification.

    Attributes:
        number: the number being classified.
        label: classification result written by the classify node.
    """

    number: int = 0
    label: str = ""


def start(state: State) -> Command:
    """Sets the number and routes directly to the classifier.

    Args:
        state: current graph state (unused on the first hop).

    Returns:
        Command that updates the number and goes to "classify".
    """
    print("start: setting number to -7")
    return Command(goto="classify", update={"number": -7})


def classify(state: State) -> Command:
    """Classifies the number as positive/negative/zero and routes accordingly.

    Args:
        state: current graph state containing the number.

    Returns:
        Command carrying the label update and the next node name.
    """
    if state.number > 0:
        label, goto = "positive", "praise"
    elif state.number < 0:
        label, goto = "negative", "warn"
    else:
        label, goto = "zero", "shrug"

    print("classify: " + str(state.number) + " is " + label + " -> goto " + goto)
    return Command(goto=goto, update={"label": label})


def praise(state: State) -> Command:
    """Handles positive numbers and ends the graph.

    Args:
        state: current graph state.

    Returns:
        Command that ends the graph.
    """
    print("praise: nice positive number!")
    return Command(goto=END)


def warn(state: State) -> Command:
    """Handles negative numbers and ends the graph.

    Args:
        state: current graph state.

    Returns:
        Command that ends the graph.
    """
    print("warn: negative numbers need care")
    return Command(goto=END)


def shrug(state: State) -> Command:
    """Handles zero and ends the graph.

    Args:
        state: current graph state.

    Returns:
        Command that ends the graph.
    """
    print("shrug: zero is just zero")
    return Command(goto=END)


def build_graph():
    """Builds and compiles the Command-style graph.

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(State)
    graph.add_node("start", start)
    graph.add_node("classify", classify)
    graph.add_node("praise", praise)
    graph.add_node("warn", warn)
    graph.add_node("shrug", shrug)

    graph.add_edge(START, "start")
    # NOTE: no add_conditional_edges anywhere - each node says where to go.
    # We still register the possible destinations with plain edges so the
    # graph knows the wiring (nodes not reachable via Command are unknown).
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({})
    print("final state:", result)
