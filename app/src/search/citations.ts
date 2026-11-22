import type { Source } from "./tools";

/** Sources the answer actually cites as [n]; the first few if it cites none. */
export function citedSources(answer: string, sources: Source[]): Source[] {
  const nums = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  const cited = sources.filter((s) => nums.has(s.n));
  return cited.length ? cited : sources.slice(0, 5);
}

/** Search language: Turkish if the question looks Turkish, otherwise English. */
export function guessLang(text: string): string {
  return /[çğıöşüÇĞİÖŞÜ]|\b(ve|bir|bu|ne|nasıl|neden|için|mı|mi|değil)\b/i.test(text) ? "tr" : "en";
}
