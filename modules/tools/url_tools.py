"""URL tools: fetch a web page and turn it into clean readable text.

Uses requests + BeautifulSoup (the engine behind LangChain's WebBaseLoader,
with our own timeouts, user agent and noise removal: script/style/nav/footer
tags are stripped before extracting text).
"""

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


def fetch_url_text(url: str, max_chars: int = 0) -> str:
    """Fetches a web page and returns its readable text content.

    Args:
        url: the page URL (must start with http:// or https://).
        max_chars: optional cap on the returned text; 0 means no limit.

    Returns:
        The page's text with scripts, styles and navigation removed.

    Raises:
        ValueError: when the URL scheme is wrong or the page cannot be
            fetched/parsed.
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        raise ValueError("URL must start with http:// or https://")

    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; learning-chatbot/1.0)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    text = "\n".join(line for line in lines if line)

    if not text:
        raise ValueError("No readable text found at that URL.")

    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


@tool
def fetch_url(url: str) -> str:
    """Fetches a web page and returns its readable text content.
    Use for reading any URL, article, or documentation page the user shares.

    Args:
        url: the full page URL, starting with http:// or https://.

    Returns:
        The page's cleaned text (truncated to ~8000 characters for the
        model), or an "Error: ..." message.
    """
    try:
        return fetch_url_text(url, max_chars=8000)
    except Exception as error:
        return "Error fetching " + url + ": " + str(error)
