"""
Pro Lesson 6: create_agent - the modern agent API

create_react_agent (used in lessons and chatbot.py) is deprecated in
LangGraph 1.x and will be REMOVED in 2.0. Its successor lives in the
langchain package:

    from langchain.agents import create_agent
    agent = create_agent(model, tools, system_prompt="...")

Same engine underneath (model node + ToolNode + the tool-vs-done router -
you built all three by hand in lesson 4), with a cleaner API and a new
extension point called MIDDLEWARE: hooks that run around the model call
and every tool call (logging, guardrails, human approval, caching...).

Migration note for chatbot.py: swap the import, rename the call, change
prompt= to system_prompt=, and pass the checkpointer the same way.
"""

import os

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

from modules.llm import get_chat_model

@tool
def word_count(text: str) -> int:
    """Counts the words in a text. Use for any 'how many words' question.

    Args:
        text: the text whose words should be counted.

    Returns:
        The number of words in the text.
    """
    return len(text.split())

def build_agent():
    """Builds an agent with the modern create_agent API.

    Returns:
        A compiled agent runnable (supports invoke/stream/checkpointer).
    """
    llm = get_chat_model()

    return create_agent(
        llm,
        tools=[word_count],
        system_prompt="Use the word_count tool for any word-counting question. Be brief.",
        checkpointer=InMemorySaver(),
    )

def main() -> None:
    """Runs the modern agent: one tool-using turn plus a memory turn.

    Shows API parity with create_react_agent: same invoke input format,
    same {"messages": [...]} output, same thread config.
    """
    agent = build_agent()
    config = {"configurable": {"thread_id": "modern"}}

    print("=== turn 1: the model uses the tool ===")
    result = agent.invoke(
        {"messages": [HumanMessage(content="How many words are in 'LangGraph makes agents simple'?")]},
        config,
    )
    print("answer:", result["messages"][-1].content)

    print()
    print("=== turn 2: memory works the same way ===")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Repeat the exact number you just gave me.")]},
        config,
    )
    print("answer:", result["messages"][-1].content)

if __name__ == "__main__":
    main()
