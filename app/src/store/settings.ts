import { readJson, writeJson } from "./files";

export type Settings = {
  searchEnabled: boolean;
  /** Optional SearXNG instance (self-hosted or trusted); DuckDuckGo and Wikipedia are used otherwise. */
  searxngUrl: string;
  /** "Write like me": answers follow the user's own writing style. */
  imitateMe: boolean;
  /** Qwen3.5 thinking mode for hard questions: slower, more accurate. */
  thinking: boolean;
  activeModelId: string | null;
};

export const DEFAULT_SETTINGS: Settings = {
  searchEnabled: true,
  searxngUrl: "",
  imitateMe: false,
  thinking: false,
  activeModelId: null,
};

const FILE = "settings.json";

export const loadSettings = (): Settings => readJson(FILE, DEFAULT_SETTINGS);
export const saveSettings = (s: Settings): void => writeJson(FILE, s);
