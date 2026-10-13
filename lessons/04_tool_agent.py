"""
Lesson 4: Tool calling - the agent loop

New ideas today:
  1. A tool is just a function with type hints and a docstring
     (the docstring is what the model reads to decide when to use it)
  2. llm.bind_tools(...) tells the model which tools exist
  3. The model "calls" a tool by attaching tool_calls to its reply -
     it never runs code itself, it only asks
  4. The agent loop: model -> tool -> model -> tool -> ... -> final answer
     The ROUTER now just relays the model's own decision. That is what
     makes this an "agent": the LLM controls the flow of the graph.

We build it manually so nothing is magic. (LangGraph ships prebuilt
shortcuts for this - mentioned after the run.)
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

from modules.llm import get_chat_model

# ---------- 1. TOOLS ----------
# The @tool decorator (the one decorator LangChain requires) turns a plain
# function into a tool: type hints become the argument schema the model sees,
# the docstring becomes the description. The model only ever SEES this
# description - the actual code runs here, in our Python.

@tool
def get_word_length(word: str) -> int:
    """Returns the number of letters in a word.

    Args:
        word: the word to measure.

    Returns:
        The number of letters (characters) in the word.
    """
    return len(word)

@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers and returns the sum.

    Args:
        a: first number.
        b: second number.

    Returns:
        The sum of the two numbers.
    """
    return a + b

TOOLS = [get_word_length, add_numbers]

# ---------- 2. STATE ----------
class State(BaseModel):
    """Agent conversation state.

    Attributes:
        messages: conversation, including AI tool requests and ToolMessage
            results (append via add_messages).
    """

    messages: Annotated[list, add_messages] = []

# ---------- 3. THE LLM, NOW AWARE OF TOOLS ----------
llm = get_chat_model()
llm_with_tools = llm.bind_tools(TOOLS)

# ---------- 4. NODE: call the model ----------
def call_model(state: State):
    """Calls the tool-aware LLM and reports whether it requested tools.

    Args:
        state: current graph state with the conversation so far.

    Returns:
        Partial update appending the AI reply (maybe with tool_calls).
    """
    reply = llm_with_tools.invoke(state.messages)
    if reply.tool_calls:
        names = [tc["name"] for tc in reply.tool_calls]
        print("call_model: model requests tools " + str(names))
    else:
        print("call_model: model gives the final answer")
    return {"messages": [reply]}

# ---------- 5. NODE: run the requested tools ----------
# The model's reply carries tool_calls: a list of {name, args, id} dicts.
# We run each one and return ToolMessages. The tool_call_id is the "receipt"
# that ties each result back to the request - the API requires it.
def call_tools(state: State):
    """Executes every tool the model requested and returns ToolMessages.

    Args:
        state: current graph state; the last message carries tool_calls.

    Returns:
        Partial update appending one ToolMessage per executed tool call.
    """
    last_reply = state.messages[-1]
    results = []
    for tool_call in last_reply.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        if name == "get_word_length":
            value = get_word_length.invoke(args)
        elif name == "add_numbers":
            value = add_numbers.invoke(args)
        else:
            value = "Unknown tool: " + name

        print("call_tools: " + name + " " + str(args) + " -> " + str(value))
        results.append(ToolMessage(content=str(value), tool_call_id=tool_call["id"]))
    return {"messages": results}

# ---------- 6. THE ROUTER: relay the model's decision ----------
# Lesson 3's router ran OUR if-statement on state. This one just checks:
# did the model ask for tools? The model itself made the decision.
def route_after_model(state: State):
    """Router relaying the model's decision: run tools or finish.

    Args:
        state: current graph state; the last message is the AI reply.

    Returns:
        "call_tools" when the model requested tools, else END.
    """
    last_reply = state.messages[-1]
    if last_reply.tool_calls:
        return "call_tools"
    return END

# ---------- 7. THE GRAPH ----------
graph = StateGraph(State)
graph.add_node("call_model", call_model)
graph.add_node("call_tools", call_tools)

graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", route_after_model)
graph.add_edge("call_tools", "call_model")   # tool results go BACK to the model

app = graph.compile()

# ---------- 8. RUN ----------
# The system message also forces tool use (otherwise the model might just
# answer "10 letters, 52" from its head and we'd see no loop).
start_messages = [
    SystemMessage(content="You must use the provided tools for any counting or math. Never compute answers yourself."),
    HumanMessage(content="How many letters are in the word 'LangGraph'? Then add that number to 42."),
]

print()
print("=== PART 1: manual tool node ===")
result = app.invoke({"messages": start_messages})

print()
print("=== final answer ===")
print(result["messages"][-1].content)

# =====================================================================
# PART 2: the SAME agent, but with ToolNode - the prebuilt node.
#
# ToolNode does what our manual call_tools node did, but automatically:
# it reads the model's tool_calls, finds each function BY NAME,
# runs it, and returns ToolMessages - no if/elif written by us.
# This is the LangGraph version of the loop LangChain's AgentExecutor
# used to hide from you. The model still decides everything (bind_tools);
# ToolNode is only the automatic executor.
# Bonus: if a tool raises an error, ToolNode catches it and returns the
# error text as the ToolMessage, so the model can react instead of crashing.
# =====================================================================

def print_history(messages):
    """Prints a readable trace of tool calls and results from the history.

    Args:
        messages: the message list from a finished agent run.
    """
    for message in messages:
        if message.type == "ai" and message.tool_calls:
            for tool_call in message.tool_calls:
                print("ai requested: " + tool_call["name"] + " " + str(tool_call["args"]))
        elif message.type == "tool":
            print("tool result: " + message.content)
        else:
            print(message.type + ": " + message.content)

graph2 = StateGraph(State)
graph2.add_node("call_model", call_model)
graph2.add_node("call_tools", ToolNode(TOOLS))   # one line replaces the whole manual node

graph2.add_edge(START, "call_model")
graph2.add_conditional_edges("call_model", route_after_model)   # same router as Part 1
graph2.add_edge("call_tools", "call_model")

app2 = graph2.compile()

print()
print("=== PART 2: same agent with ToolNode (prebuilt) ===")
result2 = app2.invoke({"messages": start_messages})

print()
print("=== final answer ===")
print(result2["messages"][-1].content)

print()
print("=== what happened inside (from message history) ===")
print_history(result2["messages"])
