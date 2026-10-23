"""LLM / embeddings factories for OpenAI, Vercel AI Gateway, or OpenCode.

All paths use langchain-openai (OpenAI-compatible HTTP). Switch with env:

    LLM_PROVIDER=openai          # direct OpenAI (default fallback)
    LLM_PROVIDER=vercel          # Vercel AI Gateway
    LLM_PROVIDER=opencode        # OpenCode Console (Zen) — pay-as-you-go
    LLM_PROVIDER=opencode-go     # OpenCode Go / Go Plus subscription
    # or leave unset: auto-selects vercel when AI_GATEWAY_API_KEY is set,
    # else opencode when OPENCODE_API_KEY is set, else openai

Auth / endpoints:

    AI_GATEWAY_API_KEY=...       # vercel
    AI_GATEWAY_BASE_URL=https://ai-gateway.vercel.sh/v1            # optional override

    OPENCODE_API_KEY=...         # opencode + opencode-go (https://opencode.ai/console)
    OPENCODE_BASE_URL=...        # optional override
    #   zen default: https://opencode.ai/inference/openai/v1
    #   go default:  https://opencode.ai/inference/go/openai/v1
    OPENCODE_SESSION_ID=...      # optional; x-opencode-session header (the Go
    # gateway 400s without one — defaults to a stable per-process id)

Model IDs:

    OPENAI_MODEL=gpt-4o-mini        # openai + vercel (and opencode fallback)
    OPENCODE_MODEL=glm-5.3-flash    # opencode / opencode-go
    OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small

Vercel rewrite bare OpenAI-style names to ``openai/<name>`` (the gateway expects
``provider/model``) — full slugs like ``anthropic/claude-sonnet-4.5`` pass
through untouched. OpenCode model ids are bare (``glm-5.3-flash``,
``kimi-k2.7-code``, …) — a leading ``opencode/`` / ``opencode-go/`` / ``zen/``
slug is stripped so OpenCode-style refs work too.

Embeddings: the OpenCode gateway serves chat models only, so when the chat
provider is opencode/opencode-go embeddings fall back to direct OpenAI. Force a
different one with ``EMBEDDINGS_PROVIDER=openai|vercel``.

Usage (after ``load_dotenv()``):

    from modules.llm import get_chat_model, get_embeddings

    llm = get_chat_model()
    embeddings = get_embeddings()
"""

from __future__ import annotations

import os
import socket
import uuid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _prefer_ipv4() -> None:
    """Connect over IPv4 first (opt out with LLM_PREFER_IPV4=0).

    On hosts with broken IPv6 routing (common on WSL), httpx stalls 30-60s on
    the IPv6 attempt before falling back, which makes every LLM call feel
    hung. Sorting A records first is harmless when IPv6 does work.
    """
    if os.getenv("LLM_PREFER_IPV4", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    if getattr(socket.getaddrinfo, "_ipv4_first", False):
        return

    original = socket.getaddrinfo

    def getaddrinfo(*args, **kwargs):
        infos = original(*args, **kwargs)
        return sorted(infos, key=lambda info: 0 if info[0] == socket.AF_INET else 1)

    getaddrinfo._ipv4_first = True  # type: ignore[attr-defined]
    socket.getaddrinfo = getaddrinfo


_prefer_ipv4()

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
OPENCODE_ZEN_BASE_URL = "https://opencode.ai/inference/openai/v1"
OPENCODE_GO_BASE_URL = "https://opencode.ai/inference/go/openai/v1"

_PROVIDER_ALIASES = {
    "openai": "openai",
    "direct": "openai",
    "oai": "openai",
    "vercel": "vercel",
    "ai-gateway": "vercel",
    "ai_gateway": "vercel",
    "gateway": "vercel",
    "opencode": "opencode",
    "zen": "opencode",
    "opencode-zen": "opencode",
    "opencode_zen": "opencode",
    "opencode-go": "opencode-go",
    "opencode_go": "opencode-go",
    "opencodego": "opencode-go",
    "go": "opencode-go",
}

_OPENCODE_PROVIDERS = ("opencode", "opencode-go")


def _normalize_provider(raw: str, var: str) -> str:
    """Maps an env value onto a known provider name."""
    key = raw.strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(
            f"Unknown {var}={raw!r} — use one of: openai, vercel, opencode, opencode-go"
        )
    return _PROVIDER_ALIASES[key]


def llm_provider() -> str:
    """Returns ``openai``, ``vercel``, ``opencode`` or ``opencode-go``.

    Reads LLM_PROVIDER; when unset, auto-selects vercel if AI_GATEWAY_API_KEY
    is set, then opencode if OPENCODE_API_KEY is set, else openai.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip()
    if explicit:
        return _normalize_provider(explicit, "LLM_PROVIDER")
    if os.getenv("AI_GATEWAY_API_KEY", "").strip():
        return "vercel"
    if os.getenv("OPENCODE_API_KEY", "").strip():
        return "opencode"
    return "openai"


def embeddings_provider() -> str:
    """Provider used for embeddings (EMBEDDINGS_PROVIDER, or the chat provider).

    OpenCode's gateway serves chat models only, so opencode / opencode-go fall
    back to direct OpenAI embeddings.
    """
    explicit = os.getenv("EMBEDDINGS_PROVIDER", "").strip()
    if explicit:
        return _normalize_provider(explicit, "EMBEDDINGS_PROVIDER")
    provider = llm_provider()
    if provider in _OPENCODE_PROVIDERS:
        return "openai"
    return provider


def _as_gateway_model(model: str) -> str:
    """Ensure a ``provider/model`` slug for AI Gateway."""
    model = model.strip()
    if "/" in model:
        return model
    return f"openai/{model}"


def _as_opencode_model(model: str) -> str:
    """OpenCode model ids are bare — strip a leading ``opencode/`` style slug."""
    model = model.strip()
    for prefix in ("opencode-go/", "opencode/", "zen/"):
        if model.lower().startswith(prefix):
            return model[len(prefix):]
    return model


def _resolve_model_name(env_vars, default: str, provider: str) -> str:
    """First non-empty env var wins, then ``default``; slugs rewritten per provider."""
    name = ""
    for var in env_vars:
        name = os.getenv(var, "").strip()
        if name:
            break
    name = name or default
    if provider == "vercel":
        return _as_gateway_model(name)
    if provider in _OPENCODE_PROVIDERS:
        return _as_opencode_model(name)
    return name


def resolve_chat_model_name(default: str = "gpt-4o-mini") -> str:
    """Chat model id for the active provider (slug-rewritten when needed).

    Reads OPENCODE_MODEL for opencode/opencode-go, else OPENAI_MODEL.
    """
    provider = llm_provider()
    if provider in _OPENCODE_PROVIDERS:
        if default == "gpt-4o-mini":
            default = "glm-5.3-flash"
        return _resolve_model_name(("OPENCODE_MODEL", "OPENAI_MODEL"), default, provider)
    return _resolve_model_name(("OPENAI_MODEL",), default, provider)


def resolve_embeddings_model_name(default: str = "text-embedding-3-small") -> str:
    """Embeddings model id for the active embeddings provider.

    OpenAI-style id via OPENAI_EMBEDDINGS_MODEL for every provider — the
    OpenCode gateway has no embeddings models, so this always runs on
    openai/vercel.
    """
    return _resolve_model_name(
        ("OPENAI_EMBEDDINGS_MODEL",), default, embeddings_provider()
    )


_PROCESS_SESSION_ID = "langgraph-chatbot-" + uuid.uuid4().hex[:12]


def _opencode_session_id() -> str:
    """Stable session id sent as ``x-opencode-session`` to the OpenCode gateway."""
    return os.getenv("OPENCODE_SESSION_ID", "").strip() or _PROCESS_SESSION_ID


def _client_kwargs(provider: str, *, embeddings: bool = False) -> dict:
    """api_key + base_url (+ headers) for the given provider."""
    if provider == "vercel":
        api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=vercel (or AI Gateway auto-detect) requires "
                "AI_GATEWAY_API_KEY in the environment / .env"
            )
        return {
            "api_key": api_key,
            "base_url": os.getenv("AI_GATEWAY_BASE_URL", AI_GATEWAY_BASE_URL).rstrip("/"),
        }

    if provider in _OPENCODE_PROVIDERS:
        if embeddings:
            raise ValueError(
                "OpenCode's gateway serves chat models only — set "
                "EMBEDDINGS_PROVIDER=openai (with OPENAI_API_KEY) or vercel "
                "(with AI_GATEWAY_API_KEY) for embeddings"
            )
        api_key = os.getenv("OPENCODE_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=opencode (or opencode-go) requires "
                "OPENCODE_API_KEY in the environment / .env"
            )
        default_base = (
            OPENCODE_GO_BASE_URL if provider == "opencode-go" else OPENCODE_ZEN_BASE_URL
        )
        # the Go gateway rejects requests without x-opencode-session (400
        # MissingSessionID), so always send a stable one per process
        headers = {
            "User-Agent": "langgraph-chatbot/1.0",
            "x-opencode-session": _opencode_session_id(),
        }
        return {
            "api_key": api_key,
            "base_url": os.getenv("OPENCODE_BASE_URL", default_base).rstrip("/"),
            "default_headers": headers,
        }

    if embeddings and not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError(
            "Embeddings run on direct OpenAI but OPENAI_API_KEY is not set — "
            "OpenCode's gateway has no embeddings endpoint, so set OPENAI_API_KEY "
            "(or EMBEDDINGS_PROVIDER=vercel with AI_GATEWAY_API_KEY)"
        )
    return {}


def _with_headers(client_kwargs: dict, kwargs: dict) -> dict:
    """Caller-supplied ``default_headers`` extend (not replace) provider headers."""
    extra = kwargs.pop("default_headers", None)
    if extra:
        merged = dict(client_kwargs.get("default_headers") or {})
        merged.update(extra)
        client_kwargs = {**client_kwargs, "default_headers": merged}
    return {**client_kwargs, **kwargs}


def get_chat_model(**kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI client for the active provider.

    Pass ``model=...`` to override OPENCODE_MODEL / OPENAI_MODEL; other kwargs
    go to ChatOpenAI.
    """
    model = kwargs.pop("model", None) or resolve_chat_model_name()
    return ChatOpenAI(model=model, **_with_headers(_client_kwargs(llm_provider()), kwargs))


def get_embeddings(**kwargs) -> OpenAIEmbeddings:
    """Build OpenAIEmbeddings for the active embeddings provider.

    Pass ``model=...`` to override OPENAI_EMBEDDINGS_MODEL.
    """
    model = kwargs.pop("model", None) or resolve_embeddings_model_name()
    provider = embeddings_provider()
    return OpenAIEmbeddings(
        model=model,
        **_with_headers(_client_kwargs(provider, embeddings=True), kwargs),
    )
