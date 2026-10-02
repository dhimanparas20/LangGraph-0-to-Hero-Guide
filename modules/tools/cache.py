"""Optional Redis cache layer shared by the web tools.

The chatbot plugs in a RedisStringUtil instance at startup via set_cache().
Without Redis everything still works - these helpers just become no-ops.
"""

# The active cache (a RedisStringUtil instance) or None when disabled.
cache = None


def set_cache(redis_cache) -> None:
    """Plugs a Redis cache into the web tools.

    Args:
        redis_cache: a modules.redis_utils.RedisStringUtil instance, or
            anything with .get(key) / .set(key, value, ttl=, overwrite=).
    """
    global cache
    cache = redis_cache


def cache_get(key: str):
    """Reads a cached value.

    Args:
        key: cache key.

    Returns:
        The cached value, or None on any failure (cache is always optional).
    """
    if cache is None:
        return None
    try:
        return cache.get(key)
    except Exception:
        return None


def cache_set(key: str, value, ttl: int) -> None:
    """Stores a value in the cache with a time-to-live.

    Args:
        key: cache key.
        value: value to store.
        ttl: seconds until the entry expires.
    """
    if cache is None:
        return
    try:
        cache.set(key, value, ttl=ttl, overwrite=True)
    except Exception:
        pass
