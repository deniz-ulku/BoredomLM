"""Keyless, free web search.

On the phone every search goes out from the user's own device and IP, so the
per-IP limits of keyless services apply per user, not to us. That is what makes
"free forever" realistic: we never pay for or proxy anyone's searches.

Providers, tried in order until enough results are collected:
  1. SearXNG  - open-source metasearch; self-hosted or any instance you trust (JSON API).
  2. ddgs     - metasearch library (DuckDuckGo, Bing, Brave, Mojeek, ...), no key.
  3. Wikipedia - official API, keyless, very reliable for encyclopedic questions.

This Python module is the reference implementation (used for data generation and
tests); the mobile apps port the same logic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx

log = logging.getLogger("candid.search")

USER_AGENT = "Candid/0.1 (+on-device assistant)"


@dataclass
class Result:
    title: str
    url: str
    snippet: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_url(url: str) -> str:
    p = urlsplit(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower() or "https", host, path, p.query, ""))


class SearXNG:
    name = "searxng"

    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None):
        self.base_url = (base_url or os.environ.get("CANDID_SEARXNG_URL", "")).rstrip("/")
        self.client = client or httpx.Client(timeout=8, headers={"User-Agent": USER_AGENT})

    def available(self) -> bool:
        return bool(self.base_url)

    def search(self, query: str, n: int, lang: str) -> list[Result]:
        r = self.client.get(f"{self.base_url}/search", params={"q": query, "format": "json", "language": lang})
        r.raise_for_status()
        return [Result(x.get("title", ""), x["url"], x.get("content", ""), self.name)
                for x in r.json().get("results", [])[:n] if x.get("url")]


class DDGS:
    name = "ddgs"

    def available(self) -> bool:
        try:
            import ddgs  # noqa: F401
        except ImportError:
            return False
        return True

    def search(self, query: str, n: int, lang: str) -> list[Result]:
        from ddgs import DDGS as _DDGS

        region = {"tr": "tr-tr", "en": "us-en"}.get(lang, "wt-wt")
        rows = _DDGS(timeout=8).text(query, region=region, safesearch="off", max_results=n)
        return [Result(x.get("title", ""), x["href"], x.get("body", ""), self.name) for x in rows if x.get("href")]


class Wikipedia:
    name = "wikipedia"

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=8, headers={"User-Agent": USER_AGENT})

    def available(self) -> bool:
        return True

    def search(self, query: str, n: int, lang: str) -> list[Result]:
        wiki = lang if lang in ("tr", "en", "de", "fr", "es") else "en"
        r = self.client.get(f"https://{wiki}.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": n,
            "format": "json", "utf8": 1, "srprop": "snippet"})
        r.raise_for_status()
        out = []
        for x in r.json().get("query", {}).get("search", []):
            title = x["title"]
            snippet = x.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
            out.append(Result(title, f"https://{wiki}.wikipedia.org/wiki/{title.replace(' ', '_')}", snippet,
                              self.name))
        return out


DEFAULT_PROVIDERS = (SearXNG, DDGS, Wikipedia)


def search(query: str, n: int = 8, lang: str = "en", providers=None) -> list[Result]:
    """Collect up to n de-duplicated results, falling through providers on error or shortfall."""
    providers = providers if providers is not None else [p() for p in DEFAULT_PROVIDERS]
    seen, out = set(), []
    for p in providers:
        if len(out) >= n:
            break
        if not p.available():
            continue
        try:
            rows = p.search(query, n, lang)
        except Exception as e:  # noqa: BLE001 - any provider failure falls through to the next
            log.warning("%s failed: %s", p.name, e)
            continue
        for r in rows:
            key = normalize_url(r.url)
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out[:n]
