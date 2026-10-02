"""
Lesson 2: An LLM inside a node + the messages state with add_messages

Two new ideas today:
  1. A node can do anything - including calling a real LLM (ChatOpenAI)
  2. The add_messages "reducer": when several nodes write to the same key,
     the writes are APPENDED to the list instead of overwriting each other
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Loads OPENAI_API_KEY from the .env file into environment variables.
load_dotenv()


# 1. STATE: a chat conversation is just a list of messages.
# Annotated[list, add_messages] tells LangGraph the merge rule for this key:
# when a node returns new messages, APPEND them (don't replace the list).
# Without add_messages the default rule is "last write wins" - so the second
# node would completely erase what the first node wrote.
class State(BaseModel):
    """Chat state whose history accumulates across nodes.

    Attributes:
        messages: conversation messages; the add_messages reducer APPENDS
            node returns instead of overwriting them.
    """

    messages: Annotated[list, add_messages] = []


# The LLM is created OUTSIDE the graph, nodes just use it.
# (LangGraph itself never calls any model - the node does. LangGraph only
# moves state between nodes.)
# Model name comes from .env (OPENAI_MODEL), with gpt-4o-mini as fallback.
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


# 2. NODE 1: writes the user's question into the conversation.
def write_question(state: State):
    """Appends the user's question as a HumanMessage.

    Args:
        state: current graph state (question text is fixed for the demo).

    Returns:
        Partial update appending the human message.
    """
    question = HumanMessage(content="Tell me one fun fact about space, in one sentence.")
    return {"messages": [question]}


# 3. NODE 2: calls the LLM with the WHOLE conversation so far,
# then appends its reply to the same list.
def answer(state: State):
    """Calls the LLM with the whole conversation and appends its reply.

    Args:
        state: current graph state with the conversation so far.

    Returns:
        Partial update appending the AI reply.
    """
    reply = llm.invoke(state.messages)
    return {"messages": [reply]}


graph = StateGraph(State)
graph.add_node("write_question", write_question)
graph.add_node("answer", answer)

graph.add_edge(START, "write_question")
graph.add_edge("write_question", "answer")
graph.add_edge("answer", END)

app = graph.compile()

# State starts empty - the default [] covers that.
result = app.invoke({})

# result["messages"] holds BOTH messages (human + ai) - the add_messages effect.
# With the default "last write wins" rule, only the ai reply would survive here.
for message in result["messages"]:
    print(message.type + ": " + message.content)
