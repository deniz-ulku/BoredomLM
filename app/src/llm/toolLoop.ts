// The agent loop: generate, run any tool calls, feed results back, repeat.
// Pure logic over an injected `complete` function, so it is unit-tested without a model.

export type ToolCall = { type: "function"; id?: string; function: { name: string; arguments: string } };

export type ChatPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

export type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | ChatPart[];
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
};

export type CompletionResult = { content: string; text: string; tool_calls?: ToolCall[] | null };

export type CompleteFn = (args: {
  messages: ChatMessage[];
  tools?: readonly object[];
  /** Visible answer text so far (tool-call markup excluded), called as it streams. */
  onText?: (textSoFar: string) => void;
  /** Override the engine's thinking setting for this call. */
  thinking?: boolean;
}) => Promise<CompletionResult>;

export type LoopEvent =
  | { kind: "text"; text: string }
  | { kind: "tool"; name: string; args: string }
  | { kind: "tool_result"; name: string; chars: number };

export type LoopOptions = {
  complete: CompleteFn;
  messages: ChatMessage[];
  tools?: readonly object[];
  runTool?: (name: string, args: string) => Promise<string>;
  maxRounds?: number;
  onEvent?: (e: LoopEvent) => void;
  maxToolChars?: number;
  thinking?: boolean;
  /** Checked between rounds: true ends the loop with what was generated so far. */
  shouldStop?: () => boolean;
};

/** Returns the final answer text and the full message list including tool turns. */
export async function runChat(opts: LoopOptions): Promise<{ answer: string; messages: ChatMessage[] }> {
  const { complete, tools, runTool, onEvent } = opts;
  const maxRounds = opts.maxRounds ?? 4;
  const maxToolChars = opts.maxToolChars ?? 6000;
  const messages = [...opts.messages];

  for (let round = 0; ; round++) {
    const useTools = tools && runTool && round < maxRounds ? tools : undefined;
    const res = await complete({
      messages,
      tools: useTools,
      onText: (text) => onEvent?.({ kind: "text", text }),
      thinking: opts.thinking,
    });
    const calls = useTools && !opts.shouldStop?.() ? (res.tool_calls ?? []).filter((c) => c?.function?.name) : [];
    if (calls.length === 0) {
      const answer = (res.content || res.text || "").trim();
      messages.push({ role: "assistant", content: answer });
      return { answer, messages };
    }
    messages.push({ role: "assistant", content: res.content ?? "", tool_calls: calls });
    for (const call of calls) {
      onEvent?.({ kind: "tool", name: call.function.name, args: call.function.arguments });
      let result = await runTool!(call.function.name, call.function.arguments);
      if (result.length > maxToolChars) result = result.slice(0, maxToolChars) + " …";
      onEvent?.({ kind: "tool_result", name: call.function.name, chars: result.length });
      messages.push({ role: "tool", name: call.function.name, tool_call_id: call.id, content: result });
    }
  }
}
