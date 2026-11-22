# On-device learning

Candid's core difference: every user's copy of the model keeps training on their phone, from their own feedback, while the phone charges. Cloud assistants cannot do this at scale: per-user weights trained every night cost too much and carry too much privacy risk on a server. On the phone the hardware is the user's and idle at night.

The loop is the same one that trains Candid on the server (`candid/`), shrunk:

```
day:    corrections, kept regenerations, own messages  ->  preference pairs (on the phone)
night:  small LoRA on the frozen base, DPO + NLL        ->  candidate adapter
check:  fixed probe set, compare with yesterday          ->  keep the new adapter, or roll back
```

## 1. Signals (built: `app/src/learn/pairs.ts`)

| Signal | Pair |
|---|---|
| User edits an answer (long-press, Correct) | chosen = edited text, rejected = original |
| User regenerates, then moves on | chosen = the version they kept, rejected = each version they replaced |

Stored in `learning.json` on the phone, newest 2,000 kept, de-duplicated by (prompt, rejected). Context is the last 6 turns ending on the user turn. Whitespace-only edits are dropped; punctuation edits are kept on purpose (a "!" turned into "." is a style signal).

Planned: thumbs-down without a correction (KTO-style unpaired negative), and the user's own sent messages as an SFT signal for "write like me".

## 2. Nightly run (to build after the feasibility spike)

Conditions: charging, screen off, battery above 50%, idle. iOS `BGProcessingTask` (`requiresExternalPower`), Android `WorkManager` (`setRequiresCharging`, `setRequiresDeviceIdle`).

Batch: up to 64 new pairs plus 25% replay of older ones (`nightlyBatch`), so the adapter does not drift toward last week only.

Training, phone-sized:
- LoRA rank 8, attention and MLP projections of the language model only; vision tower and DeltaNet gates frozen (as on the server).
- Sequences cut to 512 tokens, batch 1, gradient checkpointing.
- Loss: DPO (beta 0.1) + 1.0 NLL on chosen, learning rate 2e-4, 3 epochs; reference = the frozen base with the adapter off (settings from the simulation in section 6).
- Plus 15 anchor pairs: a worked answer to a math problem with a numeric answer over the same answer cut to its final line. They differ only in showing work, so they keep "think, then answer" working whatever style the user teaches. On the phone the worked answers come from the current model on the probe problems and are kept only when correct.
- A few hundred steps at most; the run stops when the phone is unplugged and resumes the next night.

## 3. Safety check before an adapter is used

The phone keeps a small fixed probe set (shipped with the app): 30 GSM8K-style problems with numeric answers, 20 over-refusal prompts, 10 hard-limit prompts, 20 style prompts. After training:
- accuracy on the problems may not fall more than 2 of 30 versus the current adapter;
- refusals on hard-limit prompts may not fall;
- style metrics (emoji, filler, preaching) may not get worse.

Fail: the new adapter is discarded and yesterday's stays. Every adapter is versioned; Settings can roll back or reset to the shipped model.

## 4. Platforms and formats

Inference runs on llama.rn (llama.cpp, GGUF). Training needs a framework with backprop on the phone:

| | Training | Adapter at inference |
|---|---|---|
| iPhone 17 Pro Max (12 GB) | MLX Swift (Apple's framework; LoRA training works on iOS) on the 4-bit model | export the LoRA to GGUF and load it with llama.rn `lora`, or move iOS inference to MLX too |
| Galaxy Z Fold 7 (12 GB) | llama.cpp training (ggml-opt, still experimental) or ExecuTorch training | GGUF LoRA, loaded by llama.rn |

LoRA tensors are small (rank 8: roughly 10-20 MB), so converting between MLX safetensors and GGUF LoRA on the phone is cheap; the open question is engineering, not compute.

## 5. Feasibility spike (needs the Mac and both phones)

Measure, per phone:
1. Peak memory and time per training step for rank-8 LoRA on the 4-bit 4B model at 512 tokens.
2. Steps per hour while charging, and thermal throttling over an hour.
3. Whether 30-50 real pairs change behaviour measurably overnight (from the server-side simulation, `candid/personal.py`, we know what to expect).

Go / no-go per platform from those numbers. iPhone first: MLX is the most mature path.

## 6. Server-side simulation (built: `candid/personal.py`)

Before any phone work, the same loop runs on the training GPU under phone-like limits: three synthetic users, 45 corrections each, rank-8 LoRA at 512 tokens on top of Candid v1, one adapter per user. Held-out questions measure whether answers now match the user; GSM8K (200 problems, base 90.0) measures whether capability holds. Users: **terse** (all lowercase, at most two sentences, no lists), **formal_tr** (always Turkish, formal "siz"), **facts** (peanut allergy, lives in İzmir, cat Pamuk; told in an earlier conversation, not in context, so they must come from the weights).

| Setting | terse match / GSM8K | formal_tr match / GSM8K | facts match / GSM8K |
|---|---|---|---|
| lr 2e-5, NLL 0.2 (first plan) | 0 → 0 / 91.5 | 0 → 0 / 90.5 | (facts were in context) |
| lr 1e-4, NLL 1.0 | 0 → 100 / **80.0** | 0 → 73 / 93.5 | 7 → 73 / 92.5 |
| lr 2e-4, NLL 1.0 | 0 → 80 / **79.5** | 0 → 80 / 86.5 | 7 → 100 / 94.5 |
| lr 1e-4, NLL 1.0, 15 anchors | 0 → 100 / 92.5 | 0 → 53 / 91.5 | 7 → 67 / 90.0 |
| **lr 2e-4, NLL 1.0, 15 anchors** | 0 → 93 / 91.5 | 0 → 73 / 92.0 | 7 → 93 / 90.0 |

Training took 45-60 s per user on an RTX 5090 (bf16). What it showed:
- The first plan was too timid: the pairs were learned (reward accuracy 1.0) but greedy answers did not change.
- With a usable learning rate the style and the facts move in one night, but a terse user's adapter also stopped the model showing its working on math, even when the prompt asked for it (GSM8K 90 → 80). The probe gate in section 3 would have rejected that adapter.
- 15 anchor pairs remove the regression (GSM8K 90-92.5 for every user) at a small cost in style strength. lr 2e-4 with anchors is the chosen recipe (section 2).

Targets for the phone spike: the same recipe on the 4-bit model should reach at least 70% match after one night with GSM8K-style probes within 2 of 30 of the current adapter.
