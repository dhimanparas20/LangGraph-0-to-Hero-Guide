"""Redis ops tool: the LLM's own sandboxed Redis long-term memory.

The model decides which commands to run; this tool only EXECUTES them
behind two guardrails:

  1. Command whitelist - only safe data commands. FLUSHDB/FLUSHALL,
     CONFIG, EVAL, SCRIPT, DEBUG, SHUTDOWN, SELECT, AUTH, ACL, CLIENT,
     CLUSTER, MODULE, REPLICAOF and friends are rejected outright.
  2. Automatic key prefixing - every key argument is rewritten to
     REDIS_OPS:<key> before execution, so the model physically cannot
     read, write or delete anything outside its sandbox (the chatbot's
     cache under CHATBOT:CACHE:* is untouchable, for example).

Uses the shared client from modules/clients.py.
"""

from langchain_core.tools import tool

from modules.clients import r

PREFIX = "REDIS_OPS:"

# command -> which positional args are KEYS:
#   [0, 1]  = these positions are keys
#   "all"   = every arg is a key (DEL, EXISTS, MGET)
#   "even"  = args 0, 2, 4... are keys (MSET key value key value)
#   "pattern" = arg 0 is a key PATTERN (KEYS) - prefixed like a key
#   "scan"  = the arg after MATCH is a pattern (SCAN cursor MATCH p)
#   []      = no key args at all (DBSIZE)
# Anything NOT in this map is rejected - that is the whitelist.
KEY_ARGS = {
    # strings
    "GET": [0], "SET": [0], "APPEND": [0], "STRLEN": [0],
    "MGET": "all", "MSET": "even",
    "INCR": [0], "DECR": [0], "INCRBY": [0], "DECRBY": [0],
    "GETRANGE": [0], "SETRANGE": [0],
    # keys / expiry
    "DEL": "all", "EXISTS": "all", "EXPIRE": [0], "TTL": [0],
    "PERSIST": [0], "TYPE": [0], "RENAME": [0, 1],
    # hashes
    "HSET": [0], "HGET": [0], "HGETALL": [0], "HDEL": [0],
    "HLEN": [0], "HKEYS": [0], "HVALS": [0],
    # lists
    "LPUSH": [0], "RPUSH": [0], "LPOP": [0], "RPOP": [0],
    "LRANGE": [0], "LLEN": [0], "LREM": [0], "LSET": [0],
    # sets
    "SADD": [0], "SREM": [0], "SMEMBERS": [0], "SISMEMBER": [0], "SCARD": [0],
    # sorted sets
    "ZADD": [0], "ZRANGE": [0], "ZREM": [0], "ZSCORE": [0], "ZCARD": [0],
    # scanning
    "KEYS": "pattern", "SCAN": "scan", "DBSIZE": [],
}


def redis_available() -> bool:
    """Checks that the shared Redis client can reach the server.

    Returns:
        True when a PING succeeds; False when Redis is unreachable or
        not configured (the tool is then simply not offered to the agent).
    """
    try:
        return bool(r.ping())
    except Exception:
        return False


def _prefixed(key: str) -> str:
    """Returns the key with the sandbox prefix (idempotent).

    Args:
        key: the key as the model wrote it.

    Returns:
        The key guaranteed to live inside the REDIS_OPS: namespace.
    """
    return key if key.startswith(PREFIX) else PREFIX + key


def _safe_args(command: str, args: list) -> list:
    """Validates a command and rewrites its key args into the sandbox.

    Args:
        command: the upper-cased Redis command name.
        args: the remaining arguments as strings.

    Returns:
        The argument list with every key/pattern rewritten to REDIS_OPS:*.

    Raises:
        ValueError: when the command is not whitelisted or the arguments
            do not match the command's expected shape.
    """
    if command not in KEY_ARGS:
        raise ValueError(
            "Command '" + command + "' is not allowed. Allowed commands: "
            + ", ".join(sorted(KEY_ARGS))
        )

    args = [str(a) for a in args]
    spec = KEY_ARGS[command]

    if spec == "all":
        return [_prefixed(a) for a in args]
    if spec == "even":
        return [(_prefixed(a) if i % 2 == 0 else a) for i, a in enumerate(args)]
    if spec == "pattern":
        if not args:
            raise ValueError("KEYS needs a pattern argument.")
        return [_prefixed(args[0])] + args[1:]
    if spec == "scan":
        out = []
        next_is_pattern = False
        for a in args:
            if next_is_pattern:
                out.append(_prefixed(a))
                next_is_pattern = False
            else:
                out.append(a)
                if a.upper() == "MATCH":
                    next_is_pattern = True
        return out

    # positional spec like [0] or [0, 1]
    return [(_prefixed(a) if i in spec else a) for i, a in enumerate(args)]


def _stringify(result) -> str:
    """Formats a Redis reply as readable text for the model.

    Args:
        result: the raw reply from execute_command.

    Returns:
        A compact string ("(nil)" for None, plain text otherwise).
    """
    if result is None:
        return "(nil)"
    if result is True:
        return "1"
    if result is False:
        return "0"
    return str(result)


@tool
def redis_command(command: str, arguments: list) -> str:
    """Run a Redis command in your personal REDIS_OPS sandbox - your long-term memory.
    Keys are automatically namespaced under REDIS_OPS: (never include the prefix yourself).
    Allowed: GET SET DEL EXISTS EXPIRE TTL TYPE RENAME INCR DECR MGET MSET APPEND,
    HSET HGET HGETALL HDEL HKEYS, LPUSH RPUSH LRANGE LLEN LPOP RPOP,
    SADD SMEMBERS SREM, ZADD ZRANGE ZSCORE, KEYS SCAN DBSIZE.
    Blocked: FLUSHDB FLUSHALL CONFIG EVAL SCRIPT DEBUG SHUTDOWN SELECT and anything destructive.

    Args:
        command: the Redis command name, e.g. "SET".
        arguments: the command arguments in order, e.g. ["user:1", '{"name": "Paras"}', "EX", "3600"].

    Returns:
        "OK: <reply>" on success, or an "Error: ..." message describing what went wrong.

    Examples:
        store:      redis_command("SET",  ["color", "blue", "EX", "3600"])
        read:       redis_command("GET",  ["color"])
        list keys:  redis_command("KEYS", ["*"])
        hash field: redis_command("HSET", ["profile", "city", "Delhi"])
    """
    try:
        command = command.strip().upper()
        safe = _safe_args(command, arguments)
        result = r.execute_command(command, *safe)
        return "OK: " + _stringify(result)
    except ValueError as error:
        return "Error: " + str(error)
    except Exception as error:
        return "Error: redis said - " + str(error)
