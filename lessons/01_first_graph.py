"""
Lesson 1: The core of LangGraph - State, Nodes, and Edges

Every LangGraph app is built from the same three things:
  1. STATE  - a shared object that flows through the graph
  2. NODES  - plain functions that read the state and return updates
  3. EDGES  - the wiring that decides which node runs next

No LLM in this lesson on purpose - first we learn the graph itself.
State schema uses Pydantic here - LangGraph accepts TypedDict,
Pydantic BaseModel, or dataclass equally, pick whichever you like.
"""

from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END


# 1. STATE: the shape of the data that travels through the graph.
# Pydantic version: every key can have a default, every update gets
# validated, and nodes read the state with state.name (attribute style).
class State(BaseModel):
    """Shared data flowing through the graph.

    Attributes:
        name: given by the user at invoke time.
        greeting: written by the make_greeting node.
        word_count: written by the count_words node.
    """

    name: str            # given by the user at the start
    greeting: str = ""   # written by the first node (default: empty string)
    word_count: int = 0  # written by the second node


# 2. NODES: plain Python functions.
# A node receives the current state and returns ONLY the keys it wants to change.
# Reading is attribute-style now (state.name), writing is still a plain dict.
def make_greeting(state: State):
    """Builds a greeting from the user's name.

    Args:
        state: current graph state containing the name.

    Returns:
        Partial update with the new greeting.
    """
    greeting = "Hello, " + state.name + "! Welcome to LangGraph."
    return {"greeting": greeting}


def count_words(state: State):
    """Counts the words in the current greeting.

    Args:
        state: current graph state containing the greeting.

    Returns:
        Partial update with the word count.
    """
    count = len(state.greeting.split())
    return {"word_count": count}


# 3. THE GRAPH: register nodes first, then wire them up with edges.
graph = StateGraph(State)
graph.add_node("make_greeting", make_greeting)
graph.add_node("count_words", count_words)

# 4. EDGES: the order of execution. START and END are special virtual points.
graph.add_edge(START, "make_greeting")           # entry point of the graph
graph.add_edge("make_greeting", "count_words")   # after greeting -> count
graph.add_edge("count_words", END)               # after count -> we are done

# 5. COMPILE: turns the blueprint into a runnable app.
app = graph.compile()

# 6. RUN: give it the starting values, get back the final state.
result = app.invoke({"name": "Paras"})
print(result)
