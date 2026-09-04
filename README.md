# BoredomLM-1.18B

**A 1.18-billion-parameter Transformer language model, trained from scratch on a single 8 GB consumer GPU - in one Python file.**

No cloud. No pretrained weights. Just code, data, and compute.

---

## What is this?

BoredomLM is a complete, self-contained LLM project. Everything is hand-written and lives in a single file (`train.py`):

- The Transformer model (RoPE, gated MLP, weight tying, causal attention)
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
- **Validation loss + perplexity** on a held-out set
- **Live metrics** - loss, perplexity, tokens/sec, ETA, GPU memory
- **Checkpoint comparison** - evaluate every checkpoint on the same val set
- **Automatic reports** - machine-readable JSON + Markdown
- **Plots** - loss, perplexity, throughput, memory
- **Interactive console** - a menu-driven CLI

## Quick start

```bash
python train.py
```

This opens the interactive console. Or run a mode directly:

```bash
python train.py train     # pretrain (resumes from the latest checkpoint)
python train.py finetune  # instruction fine-tune
python train.py chat      # chat with the model
python train.py evaluate  # compare checkpoints
python train.py plot      # generate plots
python train.py report    # write the report
python train.py metrics   # show latest metrics
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

## License

MIT - completely open source. Fork it, modify it, build your own version, submit improvements.

## Contributing

Pull requests and issues are welcome. This is a learning project that grew into something real - if you make it better, share it back.
