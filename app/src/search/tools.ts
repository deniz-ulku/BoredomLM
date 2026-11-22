// Tools the model can call (OpenAI / Qwen function-calling format) and their executor.

import { defaultProviders, fetchPage, search, type Fetch, type SearchResult } from "./providers";

export const TOOLS = [
  {
    type: "function",
    function: {
      name: "web_search",
      description:
        "Search the web. Use it for anything that may have changed, anything specific (names, numbers, " +
        "prices, dates, versions, news) and anything you are not sure about.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query, in the language most likely to find good sources." },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "open_url",
      description: "Read the text of a web page, usually one returned by web_search.",
      parameters: { type: "object", properties: { url: { type: "string" } }, required: ["url"] },
    },
  },
] as const;

export type Source = { n: number; title: string; url: string };

export type ToolContext = {
  searxngUrl: string;
  lang: string;
  fetchImpl?: Fetch;
  /** Every page the model saw, numbered in order, so answers can cite [n]. */
  sources: Source[];
};

function addSource(ctx: ToolContext, title: string, url: string): number {
  const existing = ctx.sources.find((s) => s.url === url);
  if (existing) return existing.n;
  const n = ctx.sources.length + 1;
  ctx.sources.push({ n, title, url });
  return n;
}

export async function runTool(name: string, argsJson: string, ctx: ToolContext): Promise<string> {
  let args: Record<string, unknown>;
  try {
    args = argsJson ? (JSON.parse(argsJson) as Record<string, unknown>) : {};
  } catch {
    return `[tool error: arguments are not valid JSON: ${argsJson.slice(0, 200)}]`;
  }
  try {
    if (name === "web_search") {
      const q = String(args.query ?? "").trim();
      if (!q) return "[tool error: empty query]";
      const rows: SearchResult[] = await search(q, defaultProviders(ctx.searxngUrl, ctx.fetchImpl), 8, ctx.lang);
      if (rows.length === 0) return "No results.";
      return rows
        .map((r) => `[${addSource(ctx, r.title, r.url)}] ${r.title}\n${r.url}\n${r.snippet}`)
        .join("\n\n");
    }
    if (name === "open_url") {
      const url = String(args.url ?? "");
      if (!/^https?:\/\//.test(url)) return "[tool error: url must start with http:// or https://]";
      const page = await fetchPage(url, ctx.fetchImpl);
      return `[${addSource(ctx, page.title || url, url)}] ${page.title}\n${url}\n\n${page.text}`;
    }
    return `[unknown tool: ${name}]`;
  } catch (e) {
    return `[tool error: ${(e as Error).message}]`;
  }
}
