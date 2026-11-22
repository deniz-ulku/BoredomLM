// On-device inference through llama.rn (llama.cpp): Metal on iPhone, OpenCL/CPU on Android.

import { initLlama, type LlamaContext } from "llama.rn";

import type { ChatMessage, CompleteFn } from "./toolLoop";

export type Engine = {
  ctx: LlamaContext;
  vision: boolean;
  modelPath: string;
  complete: CompleteFn;
  stop: () => Promise<void>;
  release: () => Promise<void>;
};

export type EngineOptions = {
  /** Qwen3.5 thinking mode: slower, better on hard problems. */
  thinking: boolean;
  nCtx?: number;
  maxTokens?: number;
};

export async function loadEngine(
  modelPath: string,
  mmprojPath: string | null,
  opts: EngineOptions,
  onProgress?: (percent: number) => void,
): Promise<Engine> {
  const ctx = await initLlama(
    {
      model: modelPath,
      n_ctx: opts.nCtx ?? 16384, // cheap here: only 8 of 32 layers keep a KV cache (3:1 DeltaNet hybrid)
      n_gpu_layers: 99,
      use_mlock: false,
      ctx_shift: false, // required for images
      flash_attn_type: "auto",
      cache_type_k: "q8_0",
      cache_type_v: "q8_0",
    },
    onProgress,
  );
  let vision = false;
  if (mmprojPath) {
    vision = await ctx.initMultimodal({ path: mmprojPath, use_gpu: true });
  }

  const complete: CompleteFn = async ({ messages, tools, onText, onThinking, thinking: thinkOverride }) => {
    const thinking = thinkOverride ?? opts.thinking;
    const res = await ctx.completion(
      {
        // llama.rn passes messages to llama.cpp's OpenAI-compatible parser, which also reads
        // tool_calls / tool_call_id; its TypeScript type only lists role and content.
        messages: messages as unknown as Parameters<LlamaContext["completion"]>[0]["messages"],
        tools: tools ? [...tools] : undefined,
        tool_choice: tools ? "auto" : undefined,
        jinja: true,
        // Both are needed: llama.rn's enable_thinking defaults to true and drives the output parser,
        // while the template reads chat_template_kwargs. "auto" moves <think> blocks (also the empty
        // one the template opens with) out of content, so the user never sees the tags.
        enable_thinking: thinking,
        reasoning_format: "auto",
        n_predict: thinking ? 6144 : opts.maxTokens ?? 2048, // reasoning needs room
        temperature: thinking ? 0.6 : 0.7,
        top_p: thinking ? 0.95 : 0.8,
        top_k: 20,
        chat_template_kwargs: { enable_thinking: thinking },
      },
      (data) => {
        if (onText && typeof data.content === "string" && data.content) onText(data.content);
        else if (onThinking && data.reasoning_content) onThinking();
      },
    );
    return { content: res.content, text: res.text, tool_calls: res.tool_calls };
  };

  return {
    ctx,
    vision,
    modelPath,
    complete,
    stop: () => ctx.stopCompletion(),
    release: () => ctx.release(),
  };
}

/** Turn an app message into the chat format; images become file:// parts. */
export function toChatMessage(role: "user" | "assistant", text: string, images: string[]): ChatMessage {
  if (role === "assistant" || images.length === 0) return { role, content: text };
  return {
    role,
    content: [
      ...images.map((url) => ({ type: "image_url" as const, image_url: { url } })),
      { type: "text" as const, text },
    ],
  };
}
