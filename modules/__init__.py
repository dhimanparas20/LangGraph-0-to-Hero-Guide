"""Shared utility modules for the course apps.

Submodules (imported lazily - import what you need):
    tools           - agent tool package (math, web, datetime)
    logger          - colorized/JSON production logger (get_logger)
    datetime_parser - human datetime parsing + timezone conversion
    mongo_utils     - MongoUtil class (user-supplied)
    redis_utils     - Redis hash/string/cache utilities (user-supplied)
    llm             - ChatOpenAI / embeddings via OpenAI or Vercel AI Gateway
    clients         - shared Redis client
"""

__all__ = [
    "tools",
    "logger",
    "datetime_parser",
    "mongo_utils",
    "redis_utils",
    "llm",
    "clients",
]
