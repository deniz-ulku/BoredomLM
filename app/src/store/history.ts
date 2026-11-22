import type { Source } from "../search/tools";
import { readJsonArray, writeJson } from "./files";

export type UiMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  /** file:// URIs of attached photos (copied into app storage). */
  images: string[];
  sources?: Source[];
  /** Deep think result: how many independent solutions agreed. */
  deep?: { support: number; total: number; agreed: boolean };
  createdAt: number;
};

const FILE = "conversation.json";

export const loadConversation = (): UiMessage[] => readJsonArray<UiMessage>(FILE);
export const saveConversation = (m: UiMessage[]): void => writeJson(FILE, m);

export const newId = (): string => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
