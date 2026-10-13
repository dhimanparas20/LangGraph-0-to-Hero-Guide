"""LLM / embeddings factories for OpenAI direct or Vercel AI Gateway.

Both paths use langchain-openai (OpenAI-compatible HTTP). Switch with env:

    LLM_PROVIDER=openai          # direct OpenAI (default if no gateway key)
    LLM_PROVIDER=vercel          # Vercel AI Gateway
    # or leave unset: auto-selects vercel when AI_GATEWAY_API_KEY is set

Gateway auth / endpoint:

    AI_GATEWAY_API_KEY=...
    AI_GATEWAY_BASE_URL=https://ai-gateway.vercel.sh/v1   # optional override

Model IDs (same vars as before):

    OPENAI_MODEL=gpt-4o-mini
    OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small

When using the gateway, bare OpenAI-style names are rewritten to
``openai/<name>`` (AI Gateway expects ``provider/model``). You can also set
full gateway IDs yourself, e.g. ``openai/gpt-4o`` or ``anthropic/claude-sonnet-4.5``.

Usage (after ``load_dotenv()``):

    from modules.llm import get_chat_model, get_embeddings

    llm = get_chat_model()
    embeddings = get_embeddings()
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"


def llm_provider() -> str:
    """Returns ``vercel`` or ``openai`` based on LLM_PROVIDER / gateway key."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("vercel", "ai-gateway", "gateway", "ai_gateway"):
        return "vercel"
    if explicit in ("openai", "direct", "oai"):
        return "openai"
    if os.getenv("AI_GATEWAY_API_KEY", "").strip():
        return "vercel"
    return "openai"


def _as_gateway_model(model: str) -> str:
    """Ensure a ``provider/model`` slug for AI Gateway."""
    model = model.strip()
    if "/" in model:
        return model
    return f"openai/{model}"


def resolve_chat_model_name(default: str = "gpt-4o-mini") -> str:
    """Chat model id for the active provider (gateway-prefixed when needed)."""
    name = os.getenv("OPENAI_MODEL", default)
    if llm_provider() == "vercel":
        return _as_gateway_model(name)
    return name


def resolve_embeddings_model_name(default: str = "text-embedding-3-small") -> str:
    """Embeddings model id for the active provider."""
    name = os.getenv("OPENAI_EMBEDDINGS_MODEL", default)
    if llm_provider() == "vercel":
        return _as_gateway_model(name)
    return name


def _openai_client_kwargs() -> dict:
    """api_key + base_url when routing through Vercel AI Gateway."""
    if llm_provider() != "vercel":
        return {}
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


def get_chat_model(**kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI client for OpenAI or Vercel AI Gateway.

    Pass ``model=...`` to override OPENAI_MODEL; other kwargs go to ChatOpenAI.
    """
    model = kwargs.pop("model", None) or resolve_chat_model_name()
    return ChatOpenAI(model=model, **_openai_client_kwargs(), **kwargs)


def get_embeddings(**kwargs) -> OpenAIEmbeddings:
    """Build OpenAIEmbeddings for OpenAI or Vercel AI Gateway.

    Pass ``model=...`` to override OPENAI_EMBEDDINGS_MODEL.
    """
    model = kwargs.pop("model", None) or resolve_embeddings_model_name()
    return OpenAIEmbeddings(model=model, **_openai_client_kwargs(), **kwargs)
