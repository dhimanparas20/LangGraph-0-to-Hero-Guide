"""MCP (Model Context Protocol) tools support.

Lets the agents use tools from MCP servers - local stdio servers (uvx/uv/npx),
local HTTP servers, and remote authenticated MCPs - without writing any
server code. Powered by langchain-mcp-adapters (MultiServerMCPClient).

This course does NOT implement MCP servers; it only CONNECTS to them.

Config formats (pick per server):

    stdio (local command):
        {"command": "uvx", "transport": "stdio", "args": [...], "env": {...}}

    local/remote HTTP (streamable):
        {"url": "http://localhost:8000/mcp/", "transport": "streamable_http"}

    remote HTTP with auth:
        {"transport": "http", "url": "https://api.example.com/mcp",
         "headers": {"Authorization": "Bearer <token>"}}

Enable in .env with MCP_ENABLED=true (default off - the chatbot works
fine without any MCP server). Failures are per-server: one dead server
never blocks the others or the chatbot.

Run a self-test:  uv run modules/tools/mcp_tools.py
"""

import asyncio
import os

# =====================================================================
# MCP server configuration - add/remove servers here (or pass your own
# dict to load_mcp_tools). Commented entries are ready-to-use examples.
# =====================================================================
MCP_TOOLS = {
    # web search via a stdio server run on demand with uvx (needs uv + internet)
    "ddg-search": {
        "command": "uvx",
        "transport": "stdio",
        "args": ["duckduckgo-mcp-server"],
        "env": {
            "DDG_SAFE_SEARCH": "MODERATE",   # "STRICT", "MODERATE", "OFF"
            "DDG_REGION": "in-en",           # India (English)
        },
    },

    # ---- your local HTTP MCP server (e.g. FastMCP) -------------------
    # "real_estate_search": {
    #     "url": os.getenv("DEFAULT_MCP_SERVER_URL", "http://localhost:8000/mcp/"),
    #     "transport": "streamable_http",
    # },

    # ---- remote authenticated MCP ------------------------------------
    # "pageindex": {
    #     "transport": "http",
    #     "url": "https://api.pageindex.ai/mcp",
    #     "headers": {"Authorization": "Bearer " + os.getenv("PAGE_INDEX_API_KEY", "")},
    # },

    # ---- more stdio examples ------------------------------------------
    # "fetch": {"command": "uv", "transport": "stdio", "args": ["run", "mcp-server-fetch"]},
    # "time": {"command": "uv", "transport": "stdio", "args": ["run", "mcp-server-time"]},
    # "firecrawl-mcp": {
    #     "command": "npx",
    #     "transport": "stdio",
    #     "args": ["-y", "firecrawl-mcp"],
    #     "env": {"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY", "")},
    # },
}


def mcp_enabled() -> bool:
    """Tells whether MCP support is switched on via the environment.

    Returns:
        True when MCP_ENABLED is "true", "1" or "yes" (default: false, so
        the repo works out of the box with no MCP servers installed).
    """
    return os.getenv("MCP_ENABLED", "false").lower() in ("true", "1", "yes")


async def _load_from_server(server_name: str, server_config: dict) -> list:
    """Connects to one MCP server and returns its tools.

    Args:
        server_name: label for this server (used in logs).
        server_config: one entry in the MCP_TOOLS format documented above.

    Returns:
        A list of LangChain BaseTool objects exposed by that server.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({server_name: server_config})
    return await client.get_tools()


def _stringify_result(result) -> str:
    """Flattens MCP tool results (lists of content blocks) into plain text.

    Args:
        result: whatever the MCP tool returned.

    Returns:
        A plain string the LLM can read.
    """
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)


def _to_sync_tool(async_tool):
    """Bridges an async-only MCP tool into a sync StructuredTool.

    MCP tools are coroutine-backed; our chatbot runs a sync graph, so each
    call runs through asyncio.run on a fresh loop (verified working).

    Args:
        async_tool: the LangChain tool returned by the MCP client.

    Returns:
        A sync StructuredTool with the same name, description and schema.
    """
    from langchain_core.tools import StructuredTool

    def run_sync(**kwargs):
        return _stringify_result(asyncio.run(async_tool.ainvoke(kwargs)))

    return StructuredTool.from_function(
        func=run_sync,
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
    )


def load_mcp_tools(config: dict = None) -> list:
    """Loads tools from every configured MCP server into the agent.

    Each server is tried independently: a dead or missing server prints a
    one-line notice and is skipped - the rest keep working. Async MCP tools
    are bridged to sync so the chatbot's sync graph can call them.

    Args:
        config: optional MCP_TOOLS-style dict; defaults to MCP_TOOLS above.

    Returns:
        A flat list of (sync) LangChain tools from all reachable servers
        (may be empty when none are configured or reachable).
    """
    servers = config if config is not None else MCP_TOOLS
    all_tools = []

    for server_name, server_config in servers.items():
        try:
            tools = asyncio.run(_load_from_server(server_name, server_config))
            print("[mcp] " + server_name + ": " + str(len(tools)) + " tools -> " + str([t.name for t in tools]))
            all_tools.extend(_to_sync_tool(t) for t in tools)
        except Exception as error:
            print("[mcp] " + server_name + ": unavailable ("
                  + str(error).split("\n")[0][:80] + ")")

    return all_tools


if __name__ == "__main__":
    print("MCP enabled:", mcp_enabled())
    print("MCP servers configured:", list(MCP_TOOLS))
    print()
    load_mcp_tools()
