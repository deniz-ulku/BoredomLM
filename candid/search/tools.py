"""Tool definitions the model sees (OpenAI/Qwen function-calling format) and their executor."""

from __future__ import annotations

import json

from candid.search.fetch import fetch
from candid.search.providers import search

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Use it for anything that may have changed, anything specific "
                           "(names, numbers, prices, dates, versions) and anything you are not sure about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, in the language most likely to find good sources."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Read the text of a web page, usually one returned by web_search.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


def run_tool(name: str, arguments: dict | str, lang: str = "en") -> str:
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    try:
        if name == "web_search":
            rows = search(args["query"], n=8, lang=lang)
            return json.dumps([r.to_dict() for r in rows], ensure_ascii=False)
        if name == "open_url":
            return fetch(args["url"])
        return f"[unknown tool: {name}]"
    except Exception as e:  # noqa: BLE001 - errors go back to the model as text
        return f"[tool error: {e}]"
