# BoredomLM-1.18B

**A 1.18-billion-parameter Transformer language model, trained from scratch on a single 8 GB consumer GPU - in one Python file.**

No cloud. No pretrained weights. Just code, data, and compute.

---

## What is this?

BoredomLM is a complete, self-contained LLM project. Everything is hand-written and lives in a single file (`train.py`):

- The Transformer model (RoPE, gated MLP, weight tying, causal attention; optional multi-token prediction and differential attention)
- The data pipeline (FineWeb pretraining, Alpaca fine-tuning)
- The training loop (BF16, 8-bit AdamW, gradient checkpointing, gradient accumulation)
- Validation, metric logging, checkpoint comparison, plots, and reports

It was trained entirely from scratch on a **single 8 GB consumer GPU** - no pretrained weights, no cloud compute.

## Architecture

| Component | Value |
|---|---|
| Parameters | ~1.18B |
| Vocabulary | 50,257 (GPT-2) |
| Hidden dim | 2048 |
| Layers | 16 |
| Attention heads | 16 (head dim 128) |
| Context length | 2048 |
| Positional encoding | RoPE |
| MLP | Gated (SwiGLU-style) |
| Weight tying | Yes |

## Features

- **Pretraining** on FineWeb
- **Instruction fine-tuning** (SFT) on Alpaca
- **Multi-token prediction (MTP)** - predict the next 4 tokens, not 1 (DeepSeek-V3's trick)
- **Differential attention** - noise-cancelling attention (the Diff Transformer)
- **Validation loss + perplexity** on a held-out set
- **Live metrics** - loss, perplexity, tokens/sec, ETA, GPU memory
- **Checkpoint comparison** - evaluate every checkpoint on the same val set
- **Brain-scan dataset** - record how the model learns (gradients, attention, probes)
- **Training-trick experiments** - automatically compare architectural choices
- **Automatic reports** - machine-readable JSON + Markdown
- **Plots** - loss, perplexity, throughput, memory
- **Interactive console** - a menu-driven CLI

## Quick start

```bash
python train.py
```

This opens the interactive console. Or run a mode directly:

```bash
python train.py train       # pretrain (resumes from the latest checkpoint)
python train.py finetune    # instruction fine-tune
python train.py chat        # chat with the model
python train.py evaluate    # compare checkpoints
python train.py plot        # generate plots
python train.py report      # write the report
python train.py brainscan   # record a brain-scan snapshot
python train.py experiment  # compare training tricks
python train.py metrics     # show latest metrics
```

## Requirements

- Python 3.9+
- An NVIDIA GPU with 8 GB+ VRAM (BF16 required)
- Dependencies auto-install on first run (`torch`, `transformers`, `datasets`, `bitsandbytes`, ...)

## How it works

1. **Pretrain** on FineWeb (next-token prediction, ~1B tokens over 500k micro-steps).
2. **Fine-tune** on Alpaca (instruction -> response).
3. **Chat** with the result.

Every run logs to `results/metrics.csv`, TensorBoard events to `runs/`, and checkpoints to `checkpoints/`.

## Brain-scan dataset

`python train.py brainscan` analyzes the latest checkpoint and writes a snapshot to `results/brain_scan/`:

- per-layer **gradient norms** (which layers are learning fastest)
- per-layer **activation norms**
- per-layer **attention entropy / max probability** + downsampled attention maps
- **probe predictions** (the model's output on fixed prompts over time)
- loss / validation loss / perplexity

This is an open record of how a from-scratch LLM actually learns. See `results/brain_scan/README.md` for the full file format.

## Experiments

`python train.py experiment` trains 6 small model variants and reports which training trick wins:

| variant | what it tests |
|---|---|
| `baseline` | the default architecture |
| `mtp` | multi-token prediction (4 tokens) |
| `diff_attn` | differential attention |
| `cosine_lr` | cosine learning-rate decay |
| `no_weight_tying` | untied output head |
| `curriculum` | length-based curriculum |

Results are written to `results/experiments.json`.

## License

MIT - completely open source. Fork it, modify it, build your own version, submit improvements.

## Contributing

Pull requests and issues are welcome. This is a learning project that grew into something real - if you make it better, share it back.
