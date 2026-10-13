"""
Lesson 8: Human-in-the-loop - pausing the graph for approval

Agents eventually do risky things: send emails, spend money, delete data.
LangGraph can PAUSE the graph before a node runs, let a human inspect
what is about to happen, and then continue or stop.

New ideas:
  1. compile(interrupt_before=["node"]) - pause right BEFORE that node
  2. get_state(config).next - which node is waiting to run
  3. invoke(None, config)  - resume where it paused (None = no new input)
  4. update_state(...) with as_node - the REJECTION path: pretend the
     node ran and put our own message into state instead

Demo: an agent that sends emails via a tool. Every send needs approval.
Part 1 approves the send. Part 2 (another thread) rejects it.
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
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

from modules.llm import get_chat_model

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email to a person. Use for anything email-related.

    Args:
        to: recipient email address.
        subject: email subject line.
        body: email body text.

    Returns:
        Confirmation string (fake send - printing only, for the lesson).
    """
    # fake on purpose - printing is enough for the lesson
    print("   >>> EMAIL ACTUALLY SENT to " + to + " with subject: " + subject)
    return "Email sent to " + to + "."

TOOLS = [send_email]

class State(BaseModel):
    """Agent state for the approval demo.

    Attributes:
        messages: conversation including pending tool calls and results.
    """

    messages: Annotated[list, add_messages] = []

llm = get_chat_model()
llm_with_tools = llm.bind_tools(TOOLS)

def call_model(state: State):
    """Calls the tool-aware LLM and reports its tool requests.

    Args:
        state: current graph state with the conversation so far.

    Returns:
        Partial update appending the AI reply.
    """
    reply = llm_with_tools.invoke(state.messages)
    if reply.tool_calls:
        print("call_model: wants to call " + str([tc["name"] for tc in reply.tool_calls]))
    else:
        print("call_model: final answer")
    return {"messages": [reply]}

def route_after_model(state: State):
    """Router relaying the model's decision (tools vs done).

    Args:
        state: current state; the last message is the AI reply.

    Returns:
        "call_tools" when tools were requested, else END.
    """
    if state.messages[-1].tool_calls:
        return "call_tools"
    return END

graph = StateGraph(State)
graph.add_node("call_model", call_model)
graph.add_node("call_tools", ToolNode(TOOLS))

graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", route_after_model)
graph.add_edge("call_tools", "call_model")

# THE NEW PART: pause before the tool node runs, every time it is about to.
app = graph.compile(checkpointer=InMemorySaver(), interrupt_before=["call_tools"])

# ================= PART 1: the human APPROVES =================
print("=== PART 1: approve the email ===")
config1 = {"configurable": {"thread_id": "approval-demo"}}

result = app.invoke(
    {"messages": [
        SystemMessage(content="Always use the send_email tool for emails."),
        HumanMessage(content="Please email bob@example.com and tell him I finished LangGraph lesson 8."),
    ]},
    config1,
)
# The graph is now PAUSED: the model asked to call send_email, but call_tools
# has not run. invoke() returned early with the partial state.

state1 = app.get_state(config1)
print("graph paused. next node waiting:", state1.next)

pending_call = state1.values["messages"][-1].tool_calls[0]
print("human reviews the pending call:")
print("   tool :", pending_call["name"])
print("   args :", pending_call["args"])

# Human says yes -> resume with invoke(None, config).
# None means "no new input, just continue where you paused".
print("human decision: APPROVED")
result = app.invoke(None, config1)   # runs call_tools, then call_model, then END
print("final: " + result["messages"][-1].content)

# ================= PART 2: the human REJECTS =================
print()
print("=== PART 2: reject the email ===")
config2 = {"configurable": {"thread_id": "rejection-demo"}}

result = app.invoke(
    {"messages": [
        SystemMessage(content="Always use the send_email tool for emails."),
        HumanMessage(content="Email alice@example.com telling her I'll be 30 minutes late to our 3pm call."),
    ]},
    config2,
)
# (Fun fact from testing this lesson: when asked to email "the office is on
# fire" to the CEO, the model refused to call the tool at all - safety
# training beat the system prompt. Models can decline; here we keep it benign.)

state2 = app.get_state(config2)
print("graph paused. next node waiting:", state2.next)
pending_call = state2.values["messages"][-1].tool_calls[0]
print("human reviews the pending call:")
print("   tool :", pending_call["name"])
print("   args :", pending_call["args"])

# Human says NO. We cannot just "not run" the node - the model is waiting for
# a ToolMessage answering its tool_call. So we WRITE state ourselves:
# update_state(..., as_node="call_tools") means "record this update as if the
# call_tools node had produced it". The graph then continues after call_tools,
# and the model sees our denial message instead of a real result.
print("human decision: REJECTED")
app.update_state(
    config2,
    {"messages": [ToolMessage(
        content="DENIED by the human supervisor. The email was NOT sent. Politely explain this to the user.",
        tool_call_id=pending_call["id"],
    )]},
    as_node="call_tools",
)

result = app.invoke(None, config2)   # resumes: back to call_model with the denial
print("final: " + result["messages"][-1].content)
