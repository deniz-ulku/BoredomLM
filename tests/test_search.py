import json

import httpx

from candid.search.fetch import extract
from candid.search.providers import Result, SearXNG, Wikipedia, normalize_url, search
from candid.search.tools import TOOLS, run_tool


class Fake:
    def __init__(self, name, rows=None, fail=False, ok=True):
        self.name, self.rows, self.fail, self.ok, self.calls = name, rows or [], fail, ok, 0

    def available(self):
        return self.ok

    def search(self, q, n, lang):
        self.calls += 1
        if self.fail:
            raise RuntimeError("rate limited")
        return self.rows[:n]


def r(url, src="x"):
    return Result("t", url, "s", src)


def test_fallthrough_and_dedup():
    a = Fake("a", fail=True)
    b = Fake("b", [r("https://www.example.com/page/"), r("https://example.com/page")])
    c = Fake("c", [r("https://other.org/x")])
    out = search("q", n=5, providers=[a, b, c])
    assert [x.url for x in out] == ["https://www.example.com/page/", "https://other.org/x"]
    assert a.calls == 1


def test_stops_when_enough():
    b = Fake("b", [r(f"https://s.com/{i}") for i in range(10)])
    c = Fake("c", [r("https://never.com")])
    assert len(search("q", n=3, providers=[b, c])) == 3
    assert c.calls == 0


def test_skips_unavailable():
    off = Fake("off", ok=False)
    on = Fake("on", [r("https://a.com")])
    search("q", providers=[off, on])
    assert off.calls == 0


def test_normalize():
    assert normalize_url("HTTPS://WWW.Example.com/a/#frag") == "https://example.com/a"


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_searxng_parsing():
    def h(req):
        assert req.url.params["format"] == "json"
        return httpx.Response(200, json={"results": [{"title": "T", "url": "https://a.com", "content": "C"}]})

    out = SearXNG("https://searx.local", client=_mock_client(h)).search("q", 5, "en")
    assert out == [Result("T", "https://a.com", "C", "searxng")]


def test_wikipedia_parsing():
    def h(req):
        assert req.url.host == "tr.wikipedia.org"
        return httpx.Response(200, json={"query": {"search": [
            {"title": "Ankara", "snippet": '<span class="searchmatch">Ankara</span> başkent'}]}})

    out = Wikipedia(client=_mock_client(h)).search("ankara", 3, "tr")
    assert out[0].url == "https://tr.wikipedia.org/wiki/Ankara"
    assert out[0].snippet == "Ankara başkent"


def test_extract_fallback_strips_markup():
    text = extract("<html><script>x=1</script><body><h1>Hi</h1><p>there</p></body></html>")
    assert "x=1" not in text and "Hi" in text and "there" in text


def test_tools_schema_and_errors():
    assert {t["function"]["name"] for t in TOOLS} == {"web_search", "open_url"}
    assert run_tool("nope", "{}").startswith("[unknown tool")
    assert run_tool("open_url", json.dumps({"url": "notaurl"})).startswith("[tool error")
