"""
Pro Lesson 2: Time travel - every step of a graph is recorded

With a checkpointer attached, LangGraph saves a snapshot after EVERY step
(not just at the end). get_state_history() walks that timeline backwards,
and you can RESUME from any past snapshot - rewinding the conversation to
an earlier point and continuing differently ("branching the past").

This is how you build: debugging/auditing, "undo" buttons, A/B testing
agent decisions, and replaying production runs.
"""

from typing import Annotated

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage


class State(BaseModel):
    """State that counts turns and accumulates messages.

    Attributes:
        messages: the running conversation (append via add_messages).
        turns: how many times the step node has run.
    """

    messages: Annotated[list, add_messages] = []
    turns: int = 0


def step(state: State) -> dict:
    """Appends one AI-ish message and bumps the turn counter.

    Args:
        state: current graph state.

    Returns:
        Partial state update with a new message and the new turn count.
    """
    text = "turn " + str(state.turns + 1) + " done"
    print("  step running ->", text)
    return {"messages": [HumanMessage(content=text)], "turns": state.turns + 1}


def build_graph():
    """Builds a one-node graph with a checkpointer (required for time travel).

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(State)
    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    return graph.compile(checkpointer=InMemorySaver())


def main(app) -> None:
    """Runs the graph three times, prints the timeline, then resumes from turn 1.

    Args:
        app: the compiled graph with a checkpointer attached.

    Steps:
        1. Invoke three times on one thread (3 checkpoints accumulate).
        2. Walk get_state_history to show every saved snapshot.
        3. Resume from the FIRST snapshot on a new thread - the graph
           continues as if turns 2 and 3 never happened.
    """
    config = {"configurable": {"thread_id": "timeline"}}

    print("running 3 turns on thread 'timeline'")
    for i in range(3):
        app.invoke({"messages": [HumanMessage(content="go " + str(i + 1))]}, config)

    print()
    print("=== timeline (newest first) ===")
    oldest = None
    for snapshot in app.get_state_history(config):
        print("  checkpoint", snapshot.config["configurable"]["checkpoint_id"][:8],
              "| turns =", snapshot.values.get("turns", 0))
        oldest = snapshot.config  # loop ends at the oldest snapshot

    print()
    print("=== resuming from the OLDEST snapshot on a new thread ===")
    branch_config = {
        "configurable": {
            "thread_id": "branch-from-turn-1",               # a NEW conversation
            "checkpoint_id": oldest["configurable"]["checkpoint_id"],  # ...starting at turn 1
        }
    }
    result = app.invoke({"messages": [HumanMessage(content="branch!")]}, branch_config)
    print("branched thread now has turns =", result["turns"],
          "(the oldest snapshot is from BEFORE turn 1, so this is 1 brand-new turn)")


if __name__ == "__main__":
    main(build_graph())
