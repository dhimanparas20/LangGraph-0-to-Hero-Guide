"""
Example Project 2: Travel Planner Agent (live web data)

Domain: trip planning with real-time data.

What it shows:
  - A modern create_agent (langchain.agents) that decides by itself which
    public-API tools to call - no manual dispatch anywhere
  - Tool reuse: imports the free web tools from modules/tools.py
  - Optional Redis caching of web results (auto-detected at startup)

Run:  uv run examples/02_travel_planner.py
Deps: OpenAI key only. Internet needed for weather/distance/wiki.
      Redis optional (set REDIS_URL in .env).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# make `modules` importable when run as: uv run examples/02_travel_planner.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import tools as tool_box
from modules.tools import get_weather, get_distance, search_wikipedia

load_dotenv()


def init_cache() -> None:
    """Enables the Redis cache for web tools when REDIS_URL is live.

    Silently skips caching when Redis is not configured or unreachable.
    """
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        print("[redis] REDIS_URL not set - caching off")
        return
    try:
        from modules.redis_utils import RedisStringUtil

        tool_box.set_cache(RedisStringUtil(url=redis_url, prefix="TRAVEL:CACHE"))
        print("[redis] connected - web results will be cached")
    except Exception as error:
        print("[redis] not reachable - caching off (" + str(error).split("\n")[0][:60] + ")")


def build_agent():
    """Builds the travel planner agent over the shared web tools.

    Returns:
        A compiled agent runnable with session memory.
    """
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return create_agent(
        llm,
        tools=[get_weather, get_distance, search_wikipedia],
        system_prompt=(
            "You are a travel planner. Use the weather tool for conditions, the "
            "distance tool for how far places are, and wikipedia for background. "
            "Always use tools for facts, then summarize in a short plan."
        ),
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    """Runs a planning request that needs all three tools, then a follow-up."""
    init_cache()
    agent = build_agent()
    config = {"configurable": {"thread_id": "trip-1"}}

    print()
    print("Q: I'm in Delhi. Is it worth visiting Manali today - weather, "
          "distance from Delhi, and one fact about the place?")
    result = agent.invoke(
        {"messages": [HumanMessage(content=(
            "I'm in Delhi. Is it worth visiting Manali today - weather, "
            "distance from Delhi, and one fact about the place?"
        ))]},
        config,
    )
    print()
    print("A:", result["messages"][-1].content)

    print()
    print("Q (follow-up, same session): and remind me how far it is?")
    result = agent.invoke(
        {"messages": [HumanMessage(content="and remind me how far it is?")]},
        config,
    )
    print("A:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
