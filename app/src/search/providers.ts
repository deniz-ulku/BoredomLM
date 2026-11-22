// Keyless web search that runs on the phone. Every request leaves from the user's own
// device and IP, so the free services' per-IP limits apply per user and cost us nothing.
// Same design as candid/search in the training repo.

import { extractText, normalizeUrl, pageTitle, parseDuckDuckGo, stripTags } from "./html";

export type SearchResult = { title: string; url: string; snippet: string; source: string };
export type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

export interface Provider {
  name: string;
  available(): boolean;
  search(query: string, n: number, lang: string): Promise<SearchResult[]>;
}

const UA = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36";

async function getText(fetchImpl: Fetch, url: string, init?: RequestInit, timeoutMs = 9000): Promise<string> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetchImpl(url, { ...init, signal: ctrl.signal, headers: { "User-Agent": UA, ...(init?.headers ?? {}) } });
    if (!r.ok) throw new Error(`HTTP ${r.status} from ${new URL(url).hostname}`);
    return await r.text();
  } finally {
    clearTimeout(timer);
  }
}

export function searxng(baseUrl: string, fetchImpl: Fetch = fetch): Provider {
  const base = baseUrl.replace(/\/+$/, "");
  return {
    name: "searxng",
    available: () => base.length > 0,
    async search(query, n, lang) {
      const q = new URLSearchParams({ q: query, format: "json", language: lang });
      const data = JSON.parse(await getText(fetchImpl, `${base}/search?${q}`)) as {
        results?: { title?: string; url?: string; content?: string }[];
      };
      return (data.results ?? [])
        .filter((r) => r.url)
        .slice(0, n)
        .map((r) => ({ title: r.title ?? "", url: r.url!, snippet: r.content ?? "", source: "searxng" }));
    },
  };
}

export function duckduckgo(fetchImpl: Fetch = fetch): Provider {
  return {
    name: "duckduckgo",
    available: () => true,
    async search(query, n, lang) {
      const kl = lang === "tr" ? "tr-tr" : lang === "en" ? "us-en" : "wt-wt";
      const body = new URLSearchParams({ q: query, kl }).toString();
      const html = await getText(fetchImpl, "https://html.duckduckgo.com/html/", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body,
      });
      return parseDuckDuckGo(html).slice(0, n).map((r) => ({ ...r, source: "duckduckgo" }));
    },
  };
}

export function wikipedia(fetchImpl: Fetch = fetch): Provider {
  return {
    name: "wikipedia",
    available: () => true,
    async search(query, n, lang) {
      const wiki = ["tr", "en", "de", "fr", "es"].includes(lang) ? lang : "en";
      const q = new URLSearchParams({
        action: "query", list: "search", srsearch: query, srlimit: String(n), format: "json", utf8: "1",
        origin: "*",
      });
      const data = JSON.parse(await getText(fetchImpl, `https://${wiki}.wikipedia.org/w/api.php?${q}`)) as {
        query?: { search?: { title: string; snippet?: string }[] };
      };
      return (data.query?.search ?? []).map((r) => ({
        title: r.title,
        url: `https://${wiki}.wikipedia.org/wiki/${encodeURIComponent(r.title.replace(/ /g, "_"))}`,
        snippet: stripTags(r.snippet ?? ""),
        source: "wikipedia",
      }));
    },
  };
}

/** Up to n de-duplicated results, falling through providers on error or shortfall. */
export async function search(query: string, providers: Provider[], n = 8, lang = "en"): Promise<SearchResult[]> {
  const seen = new Set<string>();
  const out: SearchResult[] = [];
  const errors: string[] = [];
  for (const p of providers) {
    if (out.length >= n) break;
    if (!p.available()) continue;
    try {
      for (const r of await p.search(query, n, lang)) {
        const key = normalizeUrl(r.url);
        if (!seen.has(key)) {
          seen.add(key);
          out.push(r);
        }
      }
    } catch (e) {
      errors.push(`${p.name}: ${(e as Error).message}`);
    }
  }
  if (out.length === 0 && errors.length) throw new Error(`search failed (${errors.join("; ")})`);
  return out.slice(0, n);
}

export type Page = { url: string; title: string; text: string };

export async function fetchPage(url: string, fetchImpl: Fetch = fetch, maxChars = 12000): Promise<Page> {
  const html = await getText(fetchImpl, url, undefined, 12000);
  return { url, title: pageTitle(html), text: extractText(html, maxChars) };
}

export function defaultProviders(searxngUrl: string, fetchImpl: Fetch = fetch): Provider[] {
  return [searxng(searxngUrl, fetchImpl), duckduckgo(fetchImpl), wikipedia(fetchImpl)];
}
