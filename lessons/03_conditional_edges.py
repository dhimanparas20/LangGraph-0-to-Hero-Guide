"""
Lesson 3: Conditional edges - branching and loops

New ideas today:
  1. add_conditional_edges: a "router" function READS the state and
     returns the NAME of the next node - the route is decided at runtime
  2. Edges can point BACKWARDS - that's a loop
  3. Loops MUST have a stopping condition, or the graph runs forever

The app: ask for an explanation, check its word count, and if the answer
is too long, send it back with feedback and generate again (max 3 tries).
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

from modules.llm import get_chat_model

WORD_LIMIT = 15
MAX_ATTEMPTS = 3

# STATE: same state, two different merge rules.
# messages uses the add_messages reducer (append), word_count and attempts
# are plain keys with the default rule (last write wins) - exactly what we want here.
class State(BaseModel):
    """State for the self-correcting answer loop.

    Attributes:
        messages: conversation (append via add_messages; feedback travels
            here as ordinary messages).
        word_count: words in the latest answer (plain overwrite).
        attempts: check passes so far - the loop guard (plain overwrite).
    """

    messages: Annotated[list, add_messages] = []
    word_count: int = 0
    attempts: int = 0

llm = get_chat_model()

# Node 1: ask for an explanation. Deliberately open-ended, so the first
# answer will almost surely be longer than 15 words (we want to see the loop).
def ask_question(state: State):
    """Asks an open-ended question that will surely exceed the word limit.

    Args:
        state: current graph state (unused; the question is fixed).

    Returns:
        Partial update appending the question as a HumanMessage.
    """
    question = HumanMessage(
        content="Explain what a black hole is. Cover how it forms and what the event horizon is."
    )
    return {"messages": [question]}

# Node 2: call the LLM with the whole conversation so far.
# On every retry, the conversation already contains the "too long" feedback,
# so the model knows what to fix - no extra machinery needed.
def generate_answer(state: State):
    """Calls the LLM with the whole conversation and appends its answer.

    Works unchanged on retries: the conversation already contains the
    "too long" feedback by then, so the model knows what to fix.

    Args:
        state: current graph state with the conversation so far.

    Returns:
        Partial update appending the AI answer.
    """
    reply = llm.invoke(state.messages)
    print("generate_answer: " + reply.content)
    return {"messages": [reply]}

# Node 3: plain Python check, no LLM here.
def check_answer(state: State):
    """Counts answer words; too long means feedback + one more attempt.

    Args:
        state: current graph state; the last message is the AI answer.

    Returns:
        Partial update with word_count, attempts, and - when the answer is
        too long and attempts remain - a feedback HumanMessage.
    """
    last_reply = state.messages[-1].content
    count = len(last_reply.split())
    attempts = state.attempts + 1

    if count > WORD_LIMIT and attempts < MAX_ATTEMPTS:
        feedback = HumanMessage(
            content="Your answer was " + str(count) + " words. "
                    "Rewrite it in " + str(WORD_LIMIT) + " words or fewer."
        )
        print("check_answer: " + str(count) + " words - too long, sending back (attempt " + str(attempts) + ")")
        return {"messages": [feedback], "word_count": count, "attempts": attempts}

    print("check_answer: " + str(count) + " words - ok or out of attempts, finishing")
    return {"word_count": count, "attempts": attempts}

# THE ROUTER: reads the state, returns the NAME of the next node (or END).
# This is a plain function - no LLM, just an if statement on the state.
def route_after_check(state: State):
    """Router: loop back while the answer is too long and attempts remain.

    Args:
        state: current graph state with word_count and attempts.

    Returns:
        The next node name ("generate_answer") or END.
    """
    if state.word_count > WORD_LIMIT and state.attempts < MAX_ATTEMPTS:
        return "generate_answer"   # loop BACK to generate
    return END                     # special value: the graph is done

graph = StateGraph(State)
graph.add_node("ask_question", ask_question)
graph.add_node("generate_answer", generate_answer)
graph.add_node("check_answer", check_answer)

graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "generate_answer")
graph.add_edge("generate_answer", "check_answer")

# The conditional edge: after check_answer, LangGraph calls route_after_check(state)
# and goes to whatever node it returns. Compare with add_edge, where the next
# node is fixed forever at build time.
graph.add_conditional_edges("check_answer", route_after_check)

app = graph.compile()

result = app.invoke({})

print()
print("=== final conversation ===")
for message in result["messages"]:
    print(message.type + ": " + message.content)
