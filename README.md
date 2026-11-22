# Candid

An assistant that runs on your phone and answers plainly.

- **On-device.** No message limits, no photo limits, no account, works offline. It costs us nothing per message, so there is nothing to ration.
- **Understands photos.** It reads what is actually in the picture, small print included, and says when part of it is unreadable instead of guessing.
- **Searches the web** when online, from your own device, using free keyless sources.
- **Plain answers.** No emojis, no filler, no lectures, no refusals except two narrow hard limits.
- **Does not assume.** If missing information would change the answer, it asks one short question. Otherwise it just answers.

## What makes it different

1. **It learns its user, on the phone.** Corrections and kept regenerations become training pairs on the device, and a small LoRA trains on them overnight while the phone charges, with a safety check before it is used. Each user ends up with their own model. Cloud assistants cannot do per-user nightly training at scale. Design: [docs/ON_DEVICE_LEARNING.md](docs/ON_DEVICE_LEARNING.md).
2. **It can think as long as a problem needs.** Extra tokens cost the phone only battery, so hard questions get several independent solutions, checked with tools and voted on.
3. **No limits, no lectures.** Unlimited photos and messages, plain answers, refusals only at two narrow hard limits.

## Model

Candid v0 is [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0, natively multimodal, 3:1 Gated DeltaNet / attention hybrid) with our own preference training on top. The training changes behaviour only; the gate in `candid/evaluate.py` rejects any run that loses more than one point on math, knowledge, Turkish or chart-reading benchmarks.

How we keep the model's intelligence intact, without abliteration or weight surgery:

1. The base model answers every training prompt twice: as it normally would, and with the Candid spec (`candid/spec.py`) as its system prompt.
2. Pairs are kept only where the spec answer is clean and actually better: regex checks for refusals, filler, preaching and emojis, plus the model judging both orders.
3. DPO with a small NLL term, LoRA on the language model only, vision tower frozen. Both answers in every pair come from the same model, so training moves behaviour, not knowledge. The DPO reference (the untouched base) anchors everything else.

Details: [docs/DESIGN.md](docs/DESIGN.md).

## Layout

```
candid/
  spec.py        behaviour spec, hard limits, judge prompt
  filters.py     refusal / filler / preaching / emoji / hard-limit checks
  sources.py     training prompt sources (public datasets, prompts only)
  generate.py    on-policy answer generation and pair selection (vLLM)
  train.py       DPO + NLL, LoRA, merge
  evaluate.py    before/after benchmarks and the release gate
  search/        keyless web search (SearXNG, ddgs, Wikipedia), page fetch, tool schema
remote/          what runs on the GPU machine: pipeline, GGUF export
vast/launch.py   supervisor: start / stop / rescue vast.ai machines as credit allows
tests/
app/             the iPhone + Android app (Expo, llama.rn); see app/README.md
```

## Running a training run

One v0 run is about 6 GPU-hours on an RTX 4090 equivalent; on the RTX 5090 offers the launcher usually picks, roughly $3 in total. It does not need that much credit at once.

The only key needed is a vast.ai API key in `VAST_API_KEY` (in a Claude cloud session it is a network secret and never appears in the repo or the chat).

```bash
pip install -e ".[launch]"
python vast/launch.py offers      # GPUs ranked by expected total job cost, not $/hour
python vast/launch.py tick        # run this on a schedule (hourly); it does everything below
python vast/launch.py status      # credit, machine, last progress line
python vast/launch.py logs 100
```

What `tick` does:
- No machine yet and credit above $0.10: rents the best offer and starts the run.
- Credit runs out: vast.ai stops the machine. Its disk keeps all progress (generation shards, a training checkpoint every 15 minutes, per-benchmark eval results).
- Credit is back: starts the same machine again and the pipeline continues where it stopped.
- The machine cannot start again within 20 minutes (its GPU was rented by someone else): rents a new machine, has vast.ai copy the old disk to it host-to-host, and destroys the old one once the new machine has the data.
- The pipeline finished or hit a code error: the machine stops itself and is labelled `-done` or `-failed`. A failed stage is retried once.
- The run is finished and fully backed up on the Hub: the stopped machine is destroyed (it would only pay disk rent) and no new one is launched.

Every stage's output is mirrored to the private Hugging Face repo under `runs/<run>/` (state, prompts, generation shards, pairs, checkpoints, adapter, evals, report, merged weights, GGUF), so a lost host costs at most the current shard or checkpoint interval. The finished model is `runs/<run>/gguf/`: Q4_K_M and Q8_0 plus the vision projector.

Extra lines in `remote/RUN` change a run: `redo=export` reruns a finished stage once (add any tag, `redo=export,2`, to repeat it); `mode=personal` runs the on-device learning simulation (`candid/personal.py`) instead of the pipeline.

The GGUF must be converted by the same llama.cpp build the app ships (llama.rn 0.12.9 bundles b10256); `remote/export_gguf.sh` pins it and a test checks the two agree. Newer converters write tensors that build cannot load.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

## License

Proprietary, all rights reserved (see LICENSE). Third-party components keep their own licenses (see NOTICE); the base model weights are Apache-2.0 (Qwen3.5).
