"""The tools package: every agent tool in one import.

Structure:
    cache.py          - optional Redis cache layer (set_cache plugs it in)
    math_tools.py     - calculator
    web_tools.py      - weather, distance, currency, wikipedia (free APIs)
    datetime_tools.py - current time, timezone conversion, durations
    url_tools.py      - fetch any web page as clean text
    file_tools.py     - sandboxed file management inside datastore/
    mcp_tools.py      - MCP server connector (stdio + http; MCP_ENABLED gate)

Usage (unchanged from the old single-file module):
    from modules import tools
    tools.get_weather.invoke({"city": "Delhi"})
    tools.set_cache(redis_cache)

    from modules.tools import calculator, get_current_time, load_mcp_tools
"""

from modules.tools.cache import set_cache
from modules.tools.math_tools import calculator
from modules.tools.web_tools import get_weather, get_distance, get_currency_rate, search_wikipedia
from modules.tools.datetime_tools import get_current_time, convert_time, time_between
from modules.tools.url_tools import fetch_url
from modules.tools.file_tools import DATASTORE_DIR, get_datastore_dir, load_file_management_tools
from modules.tools.mcp_tools import MCP_TOOLS, load_mcp_tools, mcp_enabled

__all__ = [
    # management
    "set_cache",
    # math
    "calculator",
    # web
    "get_weather",
    "get_distance",
    "get_currency_rate",
    "search_wikipedia",
    # datetime
    "get_current_time",
    "convert_time",
    "time_between",
    # url
    "fetch_url",
    # files
    "DATASTORE_DIR",
    "get_datastore_dir",
    "load_file_management_tools",
    # mcp
    "MCP_TOOLS",
    "load_mcp_tools",
    "mcp_enabled",
]
