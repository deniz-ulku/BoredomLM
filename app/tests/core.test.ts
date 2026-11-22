import { describe, expect, it } from "vitest";

import { extractText, normalizeUrl, parseDuckDuckGo } from "../src/search/html";
import { search, type Provider, wikipedia } from "../src/search/providers";
import { runTool, type ToolContext } from "../src/search/tools";
import { runChat, stripThink, type CompleteFn } from "../src/llm/toolLoop";
import { buildSystemPrompt } from "../src/llm/prompt";
import { buildStyleProfile, styleStats, usableSamples } from "../src/style/imitate";

const DDG_HTML = `
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title"><a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fqwen&amp;rut=abc">Qwen 3.5 <b>release</b></a></h2>
  <a class="result__snippet" href="x">The &quot;4B&quot; model &amp; more.</a>
</div>
<div class="result result--ad"><a class="result__a" href="https://ads.example">Ad</a></div>
<div class="result results_links"><a class="result__a" href="https://second.org/page">Second</a>
  <a class="result__snippet">Two</a></div>`;

describe("html", () => {
  it("parses DuckDuckGo results, unwraps redirects, skips ads", () => {
    expect(parseDuckDuckGo(DDG_HTML)).toEqual([
      { title: "Qwen 3.5 release", url: "https://example.com/qwen", snippet: 'The "4B" model & more.' },
      { title: "Second", url: "https://second.org/page", snippet: "Two" },
    ]);
  });

  it("extracts readable text and prefers <article>", () => {
    const body = "word ".repeat(120);
    const html = `<html><head><title>T</title><script>var x=1</script></head><body><nav>menu</nav>
      <article><h1>Head</h1><p>${body}</p><p>Second &amp; last</p></article><footer>foot</footer></body></html>`;
    const text = extractText(html);
    expect(text.startsWith("Head\n")).toBe(true);
    expect(text).toContain("Second & last");
    expect(text).not.toMatch(/menu|foot|var x/);
  });

  it("normalizes urls for de-duplication", () => {
    expect(normalizeUrl("https://WWW.Example.com/a/")).toBe(normalizeUrl("https://example.com/a"));
  });
});

const provider = (name: string, urls: string[], fail = false): Provider => ({
  name,
  available: () => true,
  search: async () => {
    if (fail) throw new Error("rate limited");
    return urls.map((url) => ({ title: url, url, snippet: "", source: name }));
  },
});

describe("search", () => {
  it("falls through failing providers and de-duplicates", async () => {
    const out = await search("q", [provider("a", [], true), provider("b", ["https://x.com/1", "https://www.x.com/1/"]),
      provider("c", ["https://y.com"])], 5);
    expect(out.map((r) => r.url)).toEqual(["https://x.com/1", "https://y.com"]);
  });

  it("reports every failure when nothing works", async () => {
    await expect(search("q", [provider("a", [], true), provider("b", [], true)])).rejects.toThrow(/a: rate.*b: rate/);
  });

  it("builds Wikipedia urls in the user's language", async () => {
    const fake = async (url: string) => {
      expect(new URL(url).hostname).toBe("tr.wikipedia.org");
      return new Response(JSON.stringify({ query: { search: [{ title: "Mustafa Kemal Atatürk", snippet: "<b>x</b>" }] } }));
    };
    const [r] = await wikipedia(fake).search("atatürk", 3, "tr");
    expect(r.url).toBe("https://tr.wikipedia.org/wiki/Mustafa_Kemal_Atat%C3%BCrk");
    expect(r.snippet).toBe("x");
  });
});

describe("tools", () => {
  const ctx = (): ToolContext => ({
    searxngUrl: "",
    lang: "en",
    sources: [],
    fetchImpl: async (url: string) =>
      url.includes("duckduckgo") ? new Response(DDG_HTML)
        : new Response("<title>Page</title><p>" + "content ".repeat(100) + "</p>"),
  });

  it("numbers sources across calls so answers can cite [n]", async () => {
    const c = ctx();
    const res = await runTool("web_search", JSON.stringify({ query: "qwen" }), c);
    expect(res).toMatch(/^\[1\] Qwen 3.5 release/);
    const page = await runTool("open_url", JSON.stringify({ url: "https://example.com/qwen" }), c);
    expect(page.startsWith("[1] Page")).toBe(true); // same url keeps its number
    expect(c.sources.map((s) => s.n)).toEqual([1, 2]);
  });

  it("returns errors as text for the model", async () => {
    expect(await runTool("open_url", '{"url":"ftp://x"}', ctx())).toMatch(/tool error/);
    expect(await runTool("web_search", "{bad json", ctx())).toMatch(/not valid JSON/);
    expect(await runTool("nope", "{}", ctx())).toMatch(/unknown tool/);
  });
});

describe("tool loop", () => {
  it("runs tool calls then returns the final answer", async () => {
    const seen: string[] = [];
    let calls = 0;
    const complete: CompleteFn = async ({ messages, onText }) => {
      calls++;
      if (calls === 1) {
        return { content: "", text: "", tool_calls: [{ type: "function", id: "c1",
          function: { name: "web_search", arguments: '{"query":"x"}' } }] };
      }
      expect(messages.at(-1)).toMatchObject({ role: "tool", tool_call_id: "c1", content: "RESULT" });
      onText?.("Done");
      return { content: "Done [1]", text: "Done [1]" };
    };
    const out = await runChat({
      complete, messages: [{ role: "user", content: "q" }], tools: [{}],
      runTool: async (name) => (seen.push(name), "RESULT"),
    });
    expect(out.answer).toBe("Done [1]");
    expect(seen).toEqual(["web_search"]);
    expect(out.messages.map((m) => m.role)).toEqual(["user", "assistant", "tool", "assistant"]);
  });

  it("stops offering tools after maxRounds", async () => {
    const offered: boolean[] = [];
    const complete: CompleteFn = async ({ tools }) => {
      offered.push(Boolean(tools));
      return tools
        ? { content: "", text: "", tool_calls: [{ type: "function", function: { name: "web_search", arguments: "{}" } }] }
        : { content: "final", text: "final" };
    };
    const out = await runChat({ complete, messages: [], tools: [{}], runTool: async () => "r", maxRounds: 2 });
    expect(offered).toEqual([true, true, false]);
    expect(out.answer).toBe("final");
  });
});

describe("prompt and style", () => {
  it("mentions search only when online and enabled", () => {
    const now = new Date("2026-11-22T10:00:00Z");
    expect(buildSystemPrompt({ now, searchEnabled: true, online: true })).toMatch(/web_search/);
    expect(buildSystemPrompt({ now, searchEnabled: true, online: false })).toMatch(/offline/);
    const noSearch = buildSystemPrompt({ now, searchEnabled: false, online: true });
    expect(noSearch).toMatch(/Today is 2026-11-22/);
    expect(noSearch).not.toMatch(/web_search|offline/);
  });

  it("lets the user's own style replace the default tone", () => {
    const now = new Date();
    expect(buildSystemPrompt({ now, searchEnabled: false, online: false })).toMatch(/No emojis/);
    const p = buildSystemPrompt({ now, searchEnabled: false, online: false, styleProfile: "STYLE" });
    expect(p).not.toMatch(/No emojis/);
    expect(p.endsWith("STYLE")).toBe(true);
  });

  it("learns from real messages only", () => {
    const msgs = ["ok", "https://x.com/very/long/link/here", "yarın akşam çıkıyor muyuz abi",
      "bence şu kırmızı olan daha iyi duruyor", "tamam ben 9 gibi orda olurum 😄", "tamam ben 9 gibi orda olurum 😄"];
    const s = usableSamples(msgs);
    expect(s).toEqual(["yarın akşam çıkıyor muyuz abi", "bence şu kırmızı olan daha iyi duruyor",
      "tamam ben 9 gibi orda olurum 😄"]);
    const st = styleStats(s);
    expect(st.lowercaseStart).toBe(1);
    expect(st.noFinalPunct).toBe(1);
    const profile = buildStyleProfile(msgs)!;
    expect(profile).toMatch(/lowercase/);
    expect(profile).toMatch(/uses emojis/);
    expect(buildStyleProfile(["only one real message here"])).toBeNull();
  });
});

import { citedSources, guessLang } from "../src/search/citations";
import { stringsFor } from "../src/ui/strings";

describe("citations and language", () => {
  const sources = [1, 2, 3, 4, 5, 6].map((n) => ({ n, title: `t${n}`, url: `https://s/${n}` }));
  it("keeps only cited sources, or the first five", () => {
    expect(citedSources("A [2] and B [5][2].", sources).map((s) => s.n)).toEqual([2, 5]);
    expect(citedSources("no citations", sources).map((s) => s.n)).toEqual([1, 2, 3, 4, 5]);
  });
  it("guesses Turkish questions", () => {
    expect(guessLang("Bu ne demek?")).toBe("tr");
    expect(guessLang("İstanbul hava durumu")).toBe("tr");
    expect(guessLang("What is this?")).toBe("en");
  });
  it("has the same keys in both languages", () => {
    expect(Object.keys(stringsFor("tr-TR")).sort()).toEqual(Object.keys(stringsFor("en-US")).sort());
    expect(stringsFor("tr-TR").send).toBe("Gönder");
  });
});

describe("thinking output", () => {
  it("never shows reasoning tags, even from a stopped generation", async () => {
    expect(stripThink("<think>\n\n</think>\n\nHi")).toBe("\n\nHi");
    expect(stripThink("<think>half a thought")).toBe("");
    const complete: CompleteFn = async ({ onThinking }) => {
      onThinking?.();
      return { content: "", text: "<think>working on it" };
    };
    const events: string[] = [];
    const out = await runChat({ complete, messages: [], onEvent: (e) => events.push(e.kind) });
    expect(out.answer).toBe("");
    expect(events).toContain("thinking");
  });
});
