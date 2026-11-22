// HTML helpers for on-device search: no DOM, no dependencies, so they run (and are tested) anywhere.

const ENTITIES: Record<string, string> = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—", hellip: "…",
  laquo: "«", raquo: "»", rsquo: "’", lsquo: "‘", rdquo: "”", ldquo: "“", copy: "©", reg: "®",
};

export function decodeEntities(s: string): string {
  return s.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e: string) => {
    if (e[0] === "#") {
      const code = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : m;
    }
    return ENTITIES[e.toLowerCase()] ?? m;
  });
}

export function stripTags(html: string): string {
  return decodeEntities(html.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}

const DROP = /<(script|style|noscript|svg|nav|footer|header|aside|form|iframe|template)\b[\s\S]*?<\/\1>/gi;
const BLOCK = /<\/?(p|div|li|h[1-6]|tr|br|section|article|blockquote|pre|table|ul|ol)\b[^>]*>/gi;

/** Readable text of a page: prefers <article> or <main>, keeps paragraph breaks. */
export function extractText(html: string, maxChars = 12000): string {
  let body = html.replace(/<!--[\s\S]*?-->/g, " ").replace(DROP, " ");
  const main = body.match(/<(article|main)\b[\s\S]*?<\/\1>/i);
  if (main && stripTags(main[0]).length > 400) body = main[0];
  const text = decodeEntities(body.replace(BLOCK, "\n").replace(/<[^>]+>/g, " "))
    .split("\n")
    .map((l) => l.replace(/[ \t\f\v]+/g, " ").trim())
    .filter((l) => l.length > 0)
    .join("\n");
  return text.length > maxChars ? text.slice(0, maxChars) + " …" : text;
}

export function pageTitle(html: string): string {
  const m = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  return m ? stripTags(m[1]) : "";
}

export type RawResult = { title: string; url: string; snippet: string };

/** Results from DuckDuckGo's no-JavaScript page (https://html.duckduckgo.com/html/). */
export function parseDuckDuckGo(html: string): RawResult[] {
  const out: RawResult[] = [];
  const starts = [...html.matchAll(/<div[^>]+class="([^"]*\bresult\b[^"]*)"/gi)];
  for (let i = 0; i < starts.length; i++) {
    if (/\bresult--ad\b/.test(starts[i][1])) continue;
    const b = html.slice(starts[i].index!, i + 1 < starts.length ? starts[i + 1].index : undefined);
    const a = b.match(/<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i);
    if (!a) continue;
    const url = unwrapDuckDuckGo(decodeEntities(a[1]));
    if (!url) continue;
    const sn = b.match(/class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/(a|td|div)>/i);
    out.push({ title: stripTags(a[2]), url, snippet: sn ? stripTags(sn[1]) : "" });
  }
  return out;
}

function unwrapDuckDuckGo(href: string): string | null {
  const full = href.startsWith("//") ? "https:" + href : href;
  const m = full.match(/[?&]uddg=([^&]+)/);
  const url = m ? decodeURIComponent(m[1]) : full;
  if (!/^https?:\/\//.test(url) || /duckduckgo\.com\/y\.js/.test(url)) return null;
  return url;
}

export function normalizeUrl(url: string): string {
  try {
    const u = new URL(url.trim());
    const host = u.hostname.toLowerCase().replace(/^www\./, "");
    const path = u.pathname.replace(/\/+$/, "") || "/";
    return `${u.protocol}//${host}${path}${u.search}`;
  } catch {
    return url.trim();
  }
}
