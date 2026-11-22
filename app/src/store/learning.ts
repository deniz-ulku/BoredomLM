import { addPair, type PreferencePair } from "../learn/pairs";
import { readJsonArray, writeJson } from "./files";

const FILE = "learning.json";

export const loadPairs = (): PreferencePair[] => readJsonArray<PreferencePair>(FILE);

export function savePair(p: PreferencePair | null): number {
  if (!p) return loadPairs().length;
  const all = addPair(loadPairs(), p);
  writeJson(FILE, all);
  return all.length;
}

export function savePairs(ps: PreferencePair[]): number {
  let all = loadPairs();
  for (const p of ps) all = addPair(all, p);
  writeJson(FILE, all);
  return all.length;
}

export const clearPairs = (): void => writeJson(FILE, []);
