"""
Lesson 10: Subgraphs - graphs inside graphs (intro to multi-agent)

A compiled graph can be added as a NODE in another graph:
    main_graph.add_node("science_bot", some_compiled_graph)
State flows in, state flows out. From the outside, a subgraph is just
a slightly fancy node. This is the seed of every multi-agent system.

Today: the SUPERVISOR pattern.
  - Two specialist mini-graphs (science expert, history expert),
    each built with its own system prompt
  - A supervisor node that routes the question to the right specialist
    (a router from lesson 3 + structured output from lesson 7)

Main graph:
  START -> supervisor -> science_bot -> END
                     |-> history_bot -> END
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

from modules.llm import get_chat_model

# ---------- shared state ----------
# Both the main graph and the subgraphs use this same State class.
# That is the requirement for plugging a subgraph in as a node:
# it must speak the same state language (here: the messages key).
class State(BaseModel):
    """State shared by the main graph AND both specialist subgraphs.

    Attributes:
        messages: conversation (append via add_messages).
        goto: which specialist node the supervisor picked.
    """

    messages: Annotated[list, add_messages] = []
    goto: str = ""   # which specialist the supervisor picked

llm = get_chat_model()

# ---------- build a specialist: each is a whole mini-graph ----------
# make_specialist returns a COMPILED GRAPH. Inside it is just one chat
# node with a specific personality, but it could be a whole RAG pipeline
# (lesson 6), an agent (lesson 4), anything.
def make_specialist(system_prompt):
    """Builds a compiled one-node mini-graph with the given personality.

    Args:
        system_prompt: the specialist persona injected as a SystemMessage.

    Returns:
        A compiled mini graph usable as a node in the main graph.
    """
    def chat(state: State):
        """Answers with the specialist persona applied.

        Args:
            state: current state with the conversation.

        Returns:
            Partial update appending the specialist's reply.
        """
        reply = llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
        print("   specialist answered")
        return {"messages": [reply]}

    mini = StateGraph(State)
    mini.add_node("chat", chat)
    mini.add_edge(START, "chat")
    mini.add_edge("chat", END)
    return mini.compile()

# ---------- the supervisor ----------
class Choice(BaseModel):
    """Structured supervisor decision.

    Attributes:
        specialist: "science" or "history".
    """

    specialist: str   # "science" or "history"

router_llm = llm.with_structured_output(Choice)

def supervisor(state: State):
    """Routes the newest question to the right specialist.

    Args:
        state: current state; the last message is the user question.

    Returns:
        Partial update with the chosen target node name in goto.
    """
    question = state.messages[-1].content
    choice = router_llm.invoke(
        "Route this question to the right specialist. "
        "Reply 'science' for space/physics/tech questions, 'history' for history questions.\n\n"
        "Question: " + question
    )
    goto = "history_bot" if "history" in choice.specialist.lower() else "science_bot"
    print("supervisor -> " + goto)
    return {"goto": goto}

def route_after_supervisor(state: State):
    """Router returning the specialist the supervisor chose.

    Args:
        state: current state holding goto.

    Returns:
        The next node name ("science_bot" or "history_bot").
    """
    return state.goto

# ---------- the main graph ----------
main_graph = StateGraph(State)
main_graph.add_node("supervisor", supervisor)
main_graph.add_node("science_bot", make_specialist("You are a space science expert. Answer in 1-2 short sentences."))
main_graph.add_node("history_bot", make_specialist("You are a world history expert. Answer in 1-2 short sentences."))

main_graph.add_edge(START, "supervisor")
main_graph.add_conditional_edges("supervisor", route_after_supervisor)
main_graph.add_edge("science_bot", END)
main_graph.add_edge("history_bot", END)

app = main_graph.compile()

# ---------- demo ----------
def ask(question):
    print()
    print("Q: " + question)
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print("A: " + result["messages"][-1].content)

ask("Why is the sky blue?")            # -> science_bot
ask("Who was Genghis Khan?")           # -> history_bot
