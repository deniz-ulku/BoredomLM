# Candid design notes

Decisions made so far and the reasons for them. Update this file when a decision changes.

## Product

Candid is a general assistant on the phone, not a homework tool. People should pick it over ChatGPT or Gemini because of what those products ration or add:

| ChatGPT / Gemini free tier | Candid |
|---|---|
| Message and photo limits | None, inference is on the device |
| Best features behind a subscription | No subscription |
| Account required | No account |
| Refusals, disclaimers, moralizing | Answers; two narrow hard limits |
| Long, padded answers | Short, plain answers |
| Guesses missing details | Asks one short question when it matters |
| Needs internet | Works offline; searches when online |

The honest weakness: a 4B model on a phone is weaker than frontier models on hard problems. v0 has to be good enough that limit-free, lecture-free and private wins anyway.

"Imitate me" (answers in the user's own writing style) is a setting, implemented later as a per-user LoRA or prompt profile.

## Base model

Qwen3.5-4B, chosen for:
- Apache-2.0, no use restrictions (Gemma's terms conflict with "answer everything").
- Native image understanding, strong on document and chart benchmarks for its size.
- 3:1 Gated DeltaNet / full-attention hybrid: memory stays flat on long inputs, which matters on phones.
- Existing llama.cpp (GGUF) and LiteRT conversions, so both iOS and Android have a runtime.

Memory at 4-bit is about 3 GB including the vision projector: iPhone 15 Pro and newer, recent Samsung flagships. An 8B variant can be offered later as an option for 12 GB+ devices.

Training from scratch was dropped for v0: the budget is $50, which buys a toy model from scratch but a real product on top of a strong open base.

## Training: behaviour without losing intelligence

No abliteration. Abliteration edits weights directly and measurably costs some capability.

Instead, on-policy context distillation with DPO:
- Both answers in a pair come from the base model itself (default vs. with the spec as system prompt), so the preference signal is about behaviour, not facts or reasoning.
- LoRA on the language model only; vision tower and DeltaNet gates frozen.
- The DPO reference is the frozen base, which keeps the policy close to it.
- A small NLL term on the chosen answer keeps answers fluent.
- Off-policy image-hallucination pairs (RLAIF-V) are mixed in to strengthen "only say what is visible".

Release gate (`candid/evaluate.py`):
- gsm8k, MMLU-Pro, Global-MMLU (Turkish), ChartQA: at most 1 point drop each.
- Over-refusal on OR-Bench-Hard-1k must go down. That set is excluded from training.
- Refusal rate on hard-limit prompts must not fall.
- Emoji, filler, preaching rates and answer length are reported.

v0 trains non-thinking mode only. Thinking mode stays as the base shipped it.

## Hard limits

Candid answers everything except:
1. Sexual content involving minors.
2. Meaningful uplift toward weapons capable of mass casualties.

These are the only declines. They are required for App Store and Google Play distribution of a generative AI app, and they are the right line. In training, prompts that hit these limits get a one-line plain decline as the chosen answer, no lecture.

## Search

Free, keyless, and run from the user's device so per-IP limits apply per user:
SearXNG (if configured), then ddgs (metasearch over DuckDuckGo, Bing, Brave, Mojeek and others), then Wikipedia's API. Pages are reduced to text with trafilatura.

Risk: ddgs scrapes search engines; it can break or be blocked. Wikipedia and a self-hosted SearXNG are the stable fallbacks. The mobile apps port `candid/search`.

Phase 2 trains search behaviour: when to search, how to write queries, how to combine and cite sources, with traces generated on the GPU machine using the real tools.

## Budget

Total: $50 on vast.ai.

| Item | Estimate |
|---|---|
| v0 run (generation, two evals, training, export) on one RTX 5090 | about $3 |
| Reruns and fixes | $10-15 |
| Phase 2 (search traces, second training run) | $10-15 |
| Reserve | the rest |

Credit arrives in small top-ups (a few dollars at a time), so the run has to survive running out of money at any moment:

- **Resumable pipeline** (`candid/pipeline.py`): generation in shards, a training checkpoint every 15 minutes, eval saved per benchmark. At most one shard or one 15-minute interval is lost when the machine stops.
- **State on the machine's disk.** No third-party storage account is needed. A stopped machine keeps its disk (about $0.35/day for 80 GB at typical prices).
- **Supervisor** (`vast/launch.py tick`, hourly): starts the stopped machine once credit is above $0.10. If it cannot start within 20 minutes (its GPU was rented by someone else), rents a new machine and has vast.ai copy the disk host to host. Idle machines are stopped, broken ones destroyed.
- **GPU choice** by expected total job cost (compute speed, download speed, idle storage price), not by $/hour.
- **Code delivery** without GitHub access: the code is embedded in the instance environment (xz + base64, docstrings stripped to fit vast.ai's 32 KB env limit). If the branch exists on GitHub, that is used instead.

Remaining risk: a host that disappears for good while our machine is stopped takes the progress with it. Off-host backups need a cloud storage connection in the vast.ai account.

## Roadmap

1. v0 training run, read the report, iterate on the spec and pair selection.
2. Phase 2: search-trace training, thinking-mode check.
3. Apps: llama.cpp-based iOS and Android clients with camera, search and offline mode.
4. "Imitate me" setting.
