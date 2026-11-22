// "Think harder": spend the compute a cloud service would ration. On the phone an extra token
// costs nothing but battery, so a hard question gets several independent solutions in thinking
// mode, each able to check its arithmetic with the calculator, and the answer they agree on wins.

import { CALC_TOOL, runCalc } from "./calc";
import { runChat, type ChatMessage, type CompleteFn, type LoopEvent } from "./toolLoop";

export const DEEP_INSTRUCTION =
  "Solve this carefully. Work it out step by step, use the calculate tool for every non-trivial " +
  "calculation, check your result, and end with a last line of the form: Answer: <final answer>";

/** The final answer an attempt committed to, or null. */
export function extractFinal(text: string): string | null {
  const t = text.replace(/<think>[\s\S]*?<\/think>/g, "");
  const tagged = [...t.matchAll(/(?:final answer|answer|cevap|sonuç)\s*[:：]\s*(.+)/gi)];
  if (tagged.length) return tagged[tagged.length - 1][1].trim();
  const boxed = [...t.matchAll(/\\boxed\{([^{}]+)\}/g)];
  if (boxed.length) return boxed[boxed.length - 1][1].trim();
  return null;
}

/** Canonical form for voting: "1,250.00 TL" and "1250" vote together; text is case/punctuation-folded. */
export function normalizeAnswer(answer: string): string {
  const a = answer.replace(/\*\*|`|\$|\\\(|\\\)/g, "").trim().replace(/[.。]+$/, "");
  const nums = a.match(/-?\d[\d,]*(?:\.\d+)?/g);
  if (nums && nums.length === 1) {
    const n = Number(nums[0].replace(/,/g, ""));
    if (Number.isFinite(n)) return `#${Number(n.toPrecision(12))}`;
  }
  return a.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

export type Attempt = { text: string; answer: string | null; key: string | null };
export type Vote = { key: string | null; support: number; total: number; winner: Attempt | null };

export function vote(attempts: Attempt[]): Vote {
  const counts = new Map<string, number>();
  for (const a of attempts) if (a.key) counts.set(a.key, (counts.get(a.key) ?? 0) + 1);
  let best: string | null = null;
  for (const [k, c] of counts) if (best === null || c > counts.get(best)!) best = k;
  const winner = best === null ? attempts[0] ?? null : attempts.find((a) => a.key === best) ?? null;
  return { key: best, support: best === null ? 0 : counts.get(best)!, total: attempts.length, winner };
}

export type DeepResult = { answer: string; support: number; total: number; agreed: boolean };

export type DeepOptions = {
  complete: CompleteFn;
  messages: ChatMessage[]; // system + history + the question as the last user message
  attempts?: number;
  /** Extra tools (e.g. web search) and their runner; the calculator is always available. */
  tools?: readonly object[];
  runTool?: (name: string, args: string) => Promise<string>;
  onEvent?: (e: LoopEvent | { kind: "attempt"; index: number; total: number }) => void;
  /** Stop early once this many attempts agree (default: a strict majority of `attempts`). */
  earlyStop?: number;
};

export async function deepThink(o: DeepOptions): Promise<DeepResult> {
  const n = Math.max(1, o.attempts ?? 3);
  const earlyStop = o.earlyStop ?? Math.floor(n / 2) + 1;
  const tools = [CALC_TOOL, ...(o.tools ?? [])];
  const run = async (name: string, args: string) =>
    name === "calculate" ? runCalc(args) : o.runTool ? o.runTool(name, args) : `[unknown tool: ${name}]`;
  const msgs = withInstruction(o.messages);

  const attempts: Attempt[] = [];
  for (let i = 0; i < n; i++) {
    o.onEvent?.({ kind: "attempt", index: i + 1, total: n });
    const { answer } = await runChat({ complete: o.complete, messages: msgs, tools, runTool: run,
      onEvent: o.onEvent, maxRounds: 6, thinking: true });
    const final = extractFinal(answer);
    attempts.push({ text: answer, answer: final, key: final ? normalizeAnswer(final) : null });
    const v = vote(attempts);
    if (v.support >= earlyStop) break;
  }
  const v = vote(attempts);
  return {
    answer: v.winner?.text ?? "",
    support: v.support,
    total: v.total,
    agreed: v.support >= earlyStop,
  };
}

function withInstruction(messages: ChatMessage[]): ChatMessage[] {
  const out = [...messages];
  const last = out[out.length - 1];
  if (!last || last.role !== "user") return out;
  const add = `\n\n${DEEP_INSTRUCTION}`;
  out[out.length - 1] = typeof last.content === "string" || last.content === undefined
    ? { ...last, content: (last.content ?? "") + add }
    : { ...last, content: [...last.content, { type: "text", text: DEEP_INSTRUCTION }] };
  return out;
}
