"""Fetch a page and reduce it to readable text for the model."""

from __future__ import annotations

import httpx

from candid.search.providers import USER_AGENT

MAX_BYTES = 3_000_000


def extract(html: str, url: str | None = None) -> str:
    try:
        import trafilatura

        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True, favor_recall=True)
        if text:
            return text
    except ImportError:
        pass
    # Fallback: crude tag stripping, good enough when trafilatura is unavailable.
    import re

    html = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def fetch(url: str, max_chars: int = 12_000, client: httpx.Client | None = None) -> str:
    client = client or httpx.Client(timeout=10, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    r = client.get(url)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return f"[unsupported content type: {ctype}]"
    text = extract(r.text[:MAX_BYTES], url)
    return text[:max_chars]
