"""Shared service clients - one connection object, reused everywhere.

Import this instead of creating new connections in every module:

    from modules.clients import r
    r.get("some:key")

The Redis client connects lazily (no connection attempt at import), and
uses decode_responses=True so results come back as plain strings.
"""

import os

import redis

# Shared Redis client built from REDIS_URL (.env). Lazy-connecting:
# errors only surface on the first actual command.
r = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)
