#!/usr/bin/env python3
"""train.py — single-file, cross-platform distributed LLM trainer (~1.18B params).

Double-click it (or run `python train.py`) to TRAIN directly.
Run `python train.py chat` to chat with a trained model.

For multiple machines: edit the SETTINGS block below (or set TRAIN_* env vars).
"""
import os
import sys
import subprocess
import importlib.util

_REQUIRED = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "huggingface_hub": "huggingface_hub",
    "bitsandbytes": "bitsandbytes",
    "tqdm": "tqdm",
    "pyarrow": "pyarrow",
    "pynput": "pynput",
}


def _ensure_deps():
    """Auto-install any missing packages, then restart. Makes a fresh PC work."""
    missing = [pkg for mod, pkg in _REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print("[BOOTSTRAP] Installing missing packages:", ", ".join(missing))
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *missing])
    except Exception as e:
        print("[BOOTSTRAP] Install failed:", e)
        sys.exit(1)
    print("[BOOTSTRAP] Installed. Restarting ...")
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        sys.exit(0)


_ensure_deps()


# ----------------------------------------------------------------------
# SETTINGS (edit these for multi-machine training, or use TRAIN_* env vars)
# ----------------------------------------------------------------------
RANK = 0            # 0 = this PC is the HOST; 1, 2, 3... = this PC is a WORKER
WORLD_SIZE = 1      # TOTAL number of PCs (e.g. 2 = host + 1 worker)
HOST_ADDR = ""      # WORKER only: the HOST's address, e.g. "192.168.1.50:8765"
SYNC_PORT = 8765    # port for weight syncing (same on ALL PCs)
SYNC_EVERY = 2000   # steps between weight syncs
STOP_PRESSES = 5    # F12 presses to stop training
DATA_DIR = os.path.expanduser("~/.cache/local_llm_dataset")  # dataset cache location

import glob
import time
import io
import argparse
import warnings
import threading
import urllib.request
import urllib.error
import urllib.parse
import http.server
import math
import json
import csv
import random
import socket
import platform
import datetime
from collections import deque
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.checkpoint import checkpoint

from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import snapshot_download, list_repo_files
import pyarrow.parquet as pq
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


# ----------------------------------------------------------------------
# STOP WATCHER (F12 x N, or a STOP file, or Ctrl+C)
# ----------------------------------------------------------------------
_stop_event = threading.Event()


def _start_stop_watcher():
    def watch_file():
        while not _stop_event.is_set():
            if os.path.exists("STOP"):
                print("\n[INFO] STOP file detected — stopping gracefully.")
                _stop_event.set()
                break
            time.sleep(1.0)

    threading.Thread(target=watch_file, daemon=True).start()

    try:
        from pynput import keyboard
        counter = {"n": 0}

        def on_press(key):
            if key == keyboard.Key.f12:
                counter["n"] += 1
                print(f"\n[INFO] F12 pressed {counter['n']}/{STOP_PRESSES}")
                if counter["n"] >= STOP_PRESSES:
                    print("[INFO] Stop requested — saving and exiting ...")
                    _stop_event.set()
                    return False

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        print(f"[INFO] Stop: press F12 {STOP_PRESSES}x, or create a 'STOP' file, or Ctrl+C.")
    except Exception as e:
        print(f"[WARN] F12 hotkey unavailable ({e}); use Ctrl+C or a 'STOP' file.")


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
class Config:
    # Model (~1.18B params; fits 8GB VRAM with bf16 + 8-bit optimizer)
    vocab_size = 50257            # GPT-2 vocab (matches the gpt2 tokenizer)
    dim = 2048
    n_layers = 16
    n_heads = 16                  # head_dim = 128
    seq_len = 2048

    # Training
    batch_size = 1
    grad_accum_steps = 128        # effective batch size = 128
    learning_rate = 3e-4
    weight_decay = 0.1
    warmup_steps = 2000
    total_steps = 500000

    # Data (FineWeb: huge web text — pick size via dataset_config)
    dataset_name = "HuggingFaceFW/fineweb"
    dataset_config = "sample-100BT"   # ~100B tokens. Also: "sample-10BT" (~10B)
    dataset_split = "train"
    tokenizer_name = "gpt2"
    pre_download = True           # download the dataset once so training isn't network-bound

    # Paths / logging
    save_dir = "./checkpoints"
    log_interval = 50
    save_interval = 1000
    keep_checkpoints = 3           # only keep the latest N checkpoints on disk

    # Memory / speed
    gradient_checkpointing = True
    use_flash_attention = False   # SDPA works everywhere; no flash_attn needed
    compile_model = False         # torch.compile not viable on Windows (Triton cache bug)

    # Instruction fine-tuning (SFT)
    finetune_lr = 1e-4
    finetune_steps = 5000         # ~15 min; increase for a better finetune
    finetune_seq_len = 256
    finetune_batch = 1

    # Reproducibility
    seed = 42

    # Validation (held-out files; deterministic, no train/val leakage)
    validation_interval = 2000   # micro-steps between validation runs
    validation_batches = 4       # validation batches to average per run
    val_holdout = 0.01           # fraction of files reserved for validation

    # Metrics / logging
    runs_dir = "./runs"          # TensorBoard event files
    results_dir = "./results"    # CSV / JSON / reports / plots
    experiment_name = "boredomlm-1.18b"
    log_metrics = True
    log_tensorboard = True

    # Distributed backend: "legacy" (HTTP hub-and-spoke) or "ddp" (torch.distributed)
    distributed_backend = "legacy"

    # Non-essential processes to close automatically when training starts
    close_processes = [
        "OneDrive", "OneDrive.Sync.Service",
        "XboxPcApp", "GamingServices", "XboxGameBarWidgets",
        "chrome", "msedge", "firefox", "opera",
        "Spotify", "Discord", "Steam",
    ]


config = Config()


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(2).to(x.dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(2).to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.rotary = rotary
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)

        q = self.rotary(q)
        k = self.rotary(k)

        if self.config.use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                out = flash_attn_func(q, k, v, causal=True).view(B, T, C)
                return self.o_proj(out)
            except Exception:
                pass  # fall back to SDPA

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.config = config
        self.attn = Attention(config, rotary)
        self.mlp_gate = nn.Linear(config.dim, 4 * config.dim, bias=False)
        self.mlp_up = nn.Linear(config.dim, 4 * config.dim, bias=False)
        self.mlp_down = nn.Linear(4 * config.dim, config.dim, bias=False)
        self.ln1 = nn.LayerNorm(config.dim)
        self.ln2 = nn.LayerNorm(config.dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        gate = F.gelu(self.mlp_gate(self.ln2(x)))
        up = self.mlp_up(self.ln2(x))
        x = x + self.mlp_down(gate * up)
        return x


class Transformer3B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.rotary = RotaryEmbedding(config.dim // config.n_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(config, self.rotary) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=0.8, top_p=0.9,
                 do_sample=True, eos_token_id=None, min_new_tokens=20):
        """Autoregressive text generation (temperature + top-p sampling)."""
        self.eval()
        for i in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len:]
            logits = self(idx_cond)[0][:, -1, :] / max(temperature, 1e-6)
            if eos_token_id is not None and i < min_new_tokens:
                logits[:, eos_token_id] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = probs.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
            if eos_token_id is not None and i >= min_new_tokens and next_token.item() == eos_token_id:
                break
        return idx


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def get_lr(step):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def find_latest_checkpoint():
    """Return (path, step, state) of the latest VALID checkpoint, or (None, 0, None).

    Corrupted (half-written) checkpoints are detected by failing to load and skipped.
    """
    if not os.path.isdir(config.save_dir):
        return None, 0, None
    files = [f for f in os.listdir(config.save_dir)
             if f.startswith("checkpoint_") and f.endswith(".pt")]
    files = sorted(files, key=lambda f: int(f.split("_")[1].split(".")[0]), reverse=True)
    for f in files:
        path = os.path.join(config.save_dir, f)
        try:
            state = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[WARN] Skipping corrupted checkpoint {f}: {e}")
            continue
        if isinstance(state, dict) and "model" in state:
            step = state.get("step", int(f.split("_")[1].split(".")[0]))
            return path, step, state
    final = os.path.join(config.save_dir, "final_model.pt")
    if os.path.isfile(final):
        try:
            state = torch.load(final, map_location="cpu")
            return final, state.get("step", 0), state
        except Exception as e:
            print(f"[WARN] Skipping corrupted {final}: {e}")
    return None, 0, None


def save_checkpoint(model, optimizer, step, optimizer_step, tokens_seen,
                    best_val_loss=None, best_perplexity=None, training_time=0.0):
    """Atomically save a checkpoint with full experiment metadata.

    `step` is the micro-step (semantic unchanged). `optimizer_step` and
    `tokens_seen` are persisted so they resume correctly across restarts.
    """
    path = os.path.join(config.save_dir, f"checkpoint_{step}.pt")
    tmp = path + ".tmp"
    try:
        torch.save({
            "step": step,
            "optimizer_step": optimizer_step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "best_perplexity": best_perplexity,
            "training_time": training_time,
            "seed": config.seed,
            "config": _config_dict(),
            "model": (model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()),
            "optimizer": optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, path)  # atomic: never leaves a half-written .pt behind
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    print(f"\n[SAVE] Checkpoint saved -> {path} (opt_step={optimizer_step}, tokens={tokens_seen:,})")
    _prune_checkpoints()
    return path


def _prune_checkpoints():
    """Keep only the latest `config.keep_checkpoints` checkpoints (plus final_model.pt)."""
    files = [f for f in os.listdir(config.save_dir)
             if f.startswith("checkpoint_") and f.endswith(".pt")]
    files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    for f in files[:-config.keep_checkpoints]:
        try:
            os.remove(os.path.join(config.save_dir, f))
            print(f"[CLEANUP] Removed old checkpoint {f}")
        except OSError:
            pass
    for f in os.listdir(config.save_dir):
        if f.endswith(".tmp"):
            try:
                os.remove(os.path.join(config.save_dir, f))
            except OSError:
                pass


# ----------------------------------------------------------------------
# DISTRIBUTED SYNC (hub-and-spoke: one host averages all machines' weights)
# ----------------------------------------------------------------------
def _serialize_model(model):
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    return buf.getvalue()


def _average_states(states):
    keys = list(states[0].keys())
    return {k: sum(s[k] for s in states) / len(states) for k in keys}


class _SyncHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        rank = int(q.get("rank", ["-1"])[0])
        rnd = int(q.get("round", ["-1"])[0])
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        self.server.store(rank, rnd, data)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        rnd = int(q.get("round", ["-1"])[0])
        avg = self.server.average(rnd)
        if avg is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"pending")
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(avg)))
            self.end_headers()
            self.wfile.write(avg)


class _SyncServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, world_size, port):
        self.world_size = world_size
        self._received = {}
        self._averages = {}
        super().__init__(("0.0.0.0", port), _SyncHandler)

    def store(self, rank, rnd, data):
        self._received[(rnd, rank)] = data
        ranks = {r for (rr, r) in self._received if rr == rnd}
        if len(ranks) >= self.world_size:
            states = [torch.load(io.BytesIO(self._received[(rnd, r)]),
                                 map_location="cpu", weights_only=True)
                      for r in range(self.world_size)]
            avg = _average_states(states)
            buf = io.BytesIO()
            torch.save(avg, buf)
            self._averages[rnd] = buf.getvalue()
            for r in range(self.world_size):
                self._received.pop((rnd, r), None)

    def average(self, rnd):
        return self._averages.get(rnd)


def _run_host():
    server = _SyncServer(WORLD_SIZE, SYNC_PORT)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[INFO] Host coordinator listening on 0.0.0.0:{SYNC_PORT} (world_size={WORLD_SIZE})")
    return server


def _sync_once(model, server, rnd):
    """Exchange + average model weights across all machines for round `rnd`."""
    data = _serialize_model(model)
    if RANK == 0:
        server.store(0, rnd, data)
        while True:
            avg = server.average(rnd)
            if avg is not None:
                break
            if _stop_event.is_set():
                return
            time.sleep(2.0)
    else:
        url = f"http://{HOST_ADDR}/upload?rank={RANK}&round={rnd}"
        req = urllib.request.Request(url, data=data, method="POST")
        urllib.request.urlopen(req, timeout=3600)
        while True:
            try:
                resp = urllib.request.urlopen(f"http://{HOST_ADDR}/average?round={rnd}", timeout=60)
                avg = resp.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                if _stop_event.is_set():
                    return
                time.sleep(2.0)
    model.load_state_dict(torch.load(io.BytesIO(avg), map_location="cpu", weights_only=True))
    print(f"[INFO] Synced weights (round {rnd}) across {WORLD_SIZE} machine(s).")


# ----------------------------------------------------------------------
# REPRODUCIBILITY & METRICS HELPERS
# ----------------------------------------------------------------------
def _seed_everything(seed):
    """Seed Python/NumPy/PyTorch/CUDA RNGs.

    RNGs are seeded for reproducibility, but fully deterministic cuDNN is NOT
    forced (it would substantially slow training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gpu_memory():
    """Current/peak CUDA memory in GB (zeros on CPU-only systems)."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "peak": torch.cuda.max_memory_allocated() / 1e9,
    }


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _config_dict():
    """Serializable snapshot of scalar config fields (for checkpoints/reports)."""
    return {k: v for k, v in vars(config).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}


@functools.lru_cache(maxsize=1)
def _hardware_info():
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


@torch.no_grad()
def _evaluate(model, dataloader, n_batches, device):
    """Average cross-entropy loss over `n_batches` validation batches."""
    if dataloader is None or n_batches <= 0:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    it = iter(dataloader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, targets=labels)
        total += loss.item()
        n += 1
    model.train()
    return total / n if n else float("nan")


def _write_metrics_json(path, micro_step, optimizer_step, tokens_seen,
                        best_val_loss, best_perplexity, training_time):
    """Write a compact machine-readable summary (full history lives in metrics.csv)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    summary = {
        "experiment": config.experiment_name,
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "best_perplexity": best_perplexity,
        "training_time_seconds": round(training_time, 2),
        "seed": config.seed,
        "config": _config_dict(),
        "hardware": _hardware_info(),
        "updated_at": _now_iso(),
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
# ----------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------
def _close_background_processes():
    """Terminate non-essential background processes to free CPU/disk/GPU.

    Uses Windows taskkill. Processes are matched by image name (with .exe
    appended when missing). Missing processes are silently ignored. Keeps
    VS Code, the terminal, python, explorer, and system processes intact.
    """
    for name in config.close_processes:
        exe = name if name.lower().endswith(".exe") else name + ".exe"
        subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True)


def _init_distributed():
    """Initialize torch.distributed for the 'ddp' backend.

    Reads RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT from the
    environment (set by torchrun). Returns (rank, world_size, is_distributed,
    local_rank). For the default 'legacy' backend (or world_size == 1) this is
    a no-op and returns the module-level RANK / WORLD_SIZE.
    """
    if config.distributed_backend != "ddp":
        return RANK, WORLD_SIZE, False, 0
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return rank, world_size, False, local_rank
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    return rank, world_size, True, local_rank


def train(progress_callback=None):
    _seed_everything(config.seed)
    _close_background_processes()

    rank, world_size, is_dist, local_rank = _init_distributed()
    if is_dist and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("LOCAL LLM TRAINING (~1.18B params)")
    print(f"Rank {rank}/{world_size} | Device: {device}")
    print("=" * 60)
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    print(f"[INFO] Mixed precision: {amp_dtype}")

    _start_stop_watcher()
    server = None
    if world_size != 1 and config.distributed_backend == "legacy":
        if rank == 0:
            server = _run_host()
        elif not HOST_ADDR:
            print("[ERROR] world_size != 1 and rank != 0 requires HOST_ADDR (set TRAIN_HOST).")
            sys.exit(1)

    # Tokenizer (GPT-2 — no login required)
    print(f"[INFO] Loading tokenizer '{config.tokenizer_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = int(1e9)  # we don't truncate; silence length warnings

    # Download the dataset to a local directory, then read from disk so
    # training is not stalled by network reads.
    cache_key = config.dataset_name.replace("/", "_") + (f"_{config.dataset_config}" if config.dataset_config else "")
    local_dir = os.path.join(DATA_DIR, cache_key)
    if config.pre_download and not os.path.isdir(local_dir):
        print(f"[INFO] Pre-downloading '{config.dataset_name}' ({config.dataset_config}) to local disk ...")
        try:
            if config.dataset_config:
                tag = config.dataset_config.split("-")[-1]  # "sample-100BT" -> "100BT"
                patterns = [f"sample/{tag}/*", f"{config.dataset_config}/*", f"{tag}/*"]
            else:
                patterns = "*.parquet"
            snapshot_download(
                repo_id=config.dataset_name,
                repo_type="dataset",
                allow_patterns=patterns,
                local_dir=local_dir,
            )
            print("[INFO] Dataset downloaded locally.")
        except Exception as e:
            print(f"[WARN] Pre-download failed ({e}); will stream from network instead.")

    all_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))
    all_files = [f for f in all_files
                 if not any(k in os.path.basename(f).lower() for k in ("test", "val", "valid"))]
    # Each machine reads a disjoint slice so no token is learned twice.
    if world_size != 1:
        all_files = all_files[rank::world_size]

    def _make_blocks(files):
        class _Blocks(IterableDataset):
            def __iter__(self):
                buf = []
                for f in files:
                    pf = pq.ParquetFile(f)
                    for batch in pf.iter_batches(batch_size=1024):
                        for text in batch["text"].to_pylist():
                            buf.extend(tokenizer(text or " ")["input_ids"])
                            while True:
                                if len(buf) < config.seq_len + 1:
                                    break
                                block = buf[: config.seq_len + 1]
                                buf = buf[config.seq_len:]
                                yield {
                                    "input_ids": torch.tensor(block[:-1], dtype=torch.long),
                                    "labels": torch.tensor(block[1:], dtype=torch.long),
                                }
        return _Blocks()

    if all_files:
        # Deterministic file-level train/validation split (no val tokens leak into training).
        n_val = max(1, int(round(len(all_files) * config.val_holdout)))
        val_files = all_files[-n_val:] if not (len(all_files) <= n_val + 1) else []
        train_files = [f for f in all_files if f not in set(val_files)]
        print(f"[INFO] Reading {len(train_files)} train / {len(val_files)} val parquet file(s) (rank {RANK}/{WORLD_SIZE}) ...")
        train_ds = _make_blocks(train_files)
        val_ds = _make_blocks(val_files) if val_files else None
    else:
        print(f"[INFO] Loading dataset '{config.dataset_name}' (streaming) ...")
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.dataset_split,
            streaming=True,
        )

        def tokenize(example):
            return {"input_ids": tokenizer(example.get("text", "") or " ")["input_ids"]}

        dataset = dataset.map(tokenize, batched=False, remove_columns=["text"])

        class _Blocks(IterableDataset):
            def __iter__(self):
                buf = []
                for ex in dataset:
                    buf.extend(ex["input_ids"])
                    while True:
                        if len(buf) < config.seq_len + 1:
                            break
                        block = buf[: config.seq_len + 1]
                        buf = buf[config.seq_len:]
                        yield {
                            "input_ids": torch.tensor(block[:-1], dtype=torch.long),
                            "labels": torch.tensor(block[1:], dtype=torch.long),
                        }

        train_ds = _Blocks()
        val_ds = None
        print("[WARN] Streaming mode has no held-out file split; validation metrics will be NaN.")

    dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=True) if val_ds is not None else None

    # Model (bf16 to fit 8GB)
    print(f"[INFO] Building model ({config.dim} dim, {config.n_layers} layers) ...")
    model = Transformer3B(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {n_params / 1e9:.2f}B")
    model = model.to(device=device, dtype=amp_dtype)

    optimizer = AdamW8bit(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, betas=(0.9, 0.95),
    )

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.runs_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    csv_path = os.path.join(config.results_dir, "metrics.csv")
    json_path = os.path.join(config.results_dir, "metrics.json")

    # Resume
    ckpt_path, start_step, state = find_latest_checkpoint()
    if ckpt_path and start_step != 0:
        print(f"[INFO] Resuming from {os.path.basename(ckpt_path)} at step {start_step:,}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_step = state.get("optimizer_step", start_step // config.grad_accum_steps)
        tokens_seen = state.get("tokens_seen", start_step * config.seq_len * config.batch_size)
        best_val_loss = state.get("best_val_loss", None)
        best_perplexity = state.get("best_perplexity", None)
        elapsed0 = state.get("training_time", 0.0)
    else:
        start_step = 0
        optimizer_step = 0
        tokens_seen = 0
        best_val_loss = None
        best_perplexity = None
        elapsed0 = 0.0
        print("[INFO] No checkpoint found — starting from scratch")

    # Free the checkpoint from CPU RAM (~4.7GB) so it doesn't cause swapping.
    state = None
    import gc
    gc.collect()

    if config.compile_model:
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile failed ({e}); continuing without it")

    if is_dist:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None
        )

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step = start_step
    running_loss = 0.0
    data_iter = iter(dataloader)
    process_start = time.time()
    stopped = False

    # Throughput / metric bookkeeping
    tok_per_step = config.seq_len * config.batch_size   # tokens processed per micro-step
    recent_tps = deque(maxlen=100)                      # moving-average tokens/sec
    last_log_time = time.time()

    writer = None
    if config.log_tensorboard and SummaryWriter is not None and rank == 0:
        try:
            writer = SummaryWriter(log_dir=os.path.join(config.runs_dir, config.experiment_name))
        except Exception as e:
            print(f"[WARN] TensorBoard unavailable ({e})")

    csv_header = ["micro_step", "optimizer_step", "train_loss", "val_loss", "perplexity",
                  "learning_rate", "tokens_seen", "tokens_per_second", "elapsed_seconds",
                  "gpu_memory_allocated", "gpu_memory_reserved", "peak_vram"]
    if config.log_metrics and not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    pbar = tqdm(total=config.total_steps, initial=start_step, desc="Training", unit="step")
    first_batch = True

    try:
        while step < config.total_steps:
            if _stop_event.is_set():
                print("\n[INFO] Stop requested — saving checkpoint ...")
                save_checkpoint(model, optimizer, step + 1, optimizer_step, tokens_seen,
                                best_val_loss, best_perplexity, elapsed0 + (time.time() - process_start))
                stopped = True
                break

            for g in optimizer.param_groups:
                g["lr"] = get_lr(step)

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if first_batch:
                print(f"[INFO] First batch loaded {tuple(batch['input_ids'].shape)} — training is running.")
                first_batch = False

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, targets=labels)
            loss = loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += loss.item() * config.grad_accum_steps

            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                optimizer_step += 1

            tokens_seen += tok_per_step
            pbar.update(1)  # move the bar every step

            if world_size != 1 and config.distributed_backend == "legacy" and (step + 1) % SYNC_EVERY == 0:
                _sync_once(model, server, (step + 1) // SYNC_EVERY)

            do_log = (step + 1) % config.log_interval == 0
            do_val = (val_loader is not None and (step + 1) % config.validation_interval == 0)

            if do_log or do_val:
                now = time.time()
                elapsed = now - process_start

                if do_log:
                    avg_loss = running_loss / config.log_interval
                    dt = now - last_log_time
                    if dt != 0.0:
                        recent_tps.append(config.log_interval * tok_per_step / dt)
                    last_log_time = now
                else:
                    avg_loss = float("nan")

                if do_val:
                    val_loss = _evaluate(model, val_loader, config.validation_batches, device)
                    perplexity = math.exp(val_loss) if (val_loss == val_loss and val_loss < 100.0) else float("nan")
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_perplexity = perplexity
                else:
                    val_loss = float("nan")
                    perplexity = float("nan")

                gpu = _gpu_memory()
                ma_tps = (sum(recent_tps) / len(recent_tps)) if recent_tps else 0.0
                cur_lr = get_lr(step)
                total_elapsed = elapsed0 + elapsed
                eta_seconds = ((config.total_steps - (step + 1)) * tok_per_step / ma_tps) if ma_tps != 0 else 0.0

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "val": f"{val_loss:.4f}" if val_loss == val_loss else "nan",
                    "ppl": f"{perplexity:.2f}" if perplexity == perplexity else "nan",
                    "lr": f"{cur_lr:.2e}",
                    "tok/s": f"{ma_tps:.0f}",
                    "eta": f"{eta_seconds/3600:.1f}h",
                })

                if progress_callback is not None:
                    progress_callback({
                        "micro_step": step + 1,
                        "optimizer_step": optimizer_step,
                        "train_loss": avg_loss,
                        "val_loss": val_loss,
                        "perplexity": perplexity,
                        "learning_rate": cur_lr,
                        "tokens_seen": tokens_seen,
                        "tokens_per_second": ma_tps,
                        "elapsed_seconds": total_elapsed,
                        "eta_seconds": eta_seconds,
                    })

                if config.log_metrics and rank == 0:
                    row = [step + 1, optimizer_step,
                           round(avg_loss, 4) if avg_loss == avg_loss else "",
                           round(val_loss, 4) if val_loss == val_loss else "",
                           round(perplexity, 4) if perplexity == perplexity else "",
                           cur_lr, tokens_seen, round(ma_tps, 2), round(total_elapsed, 2),
                           round(gpu["allocated"], 4), round(gpu["reserved"], 4), round(gpu["peak"], 4)]
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(row)

                    try:
                        _write_metrics_json(json_path, step + 1, optimizer_step, tokens_seen,
                                            best_val_loss, best_perplexity, total_elapsed)
                    except Exception as e:
                        print(f"\n[WARN] metrics.json write failed ({e})")

                    if writer is not None:
                        try:
                            writer.add_scalar("train/loss", avg_loss, step + 1)
                            writer.add_scalar("validation/loss", val_loss, step + 1)
                            writer.add_scalar("validation/perplexity", perplexity, step + 1)
                            writer.add_scalar("training/learning_rate", cur_lr, step + 1)
                            writer.add_scalar("training/tokens_per_second", ma_tps, step + 1)
                            writer.add_scalar("training/tokens_seen", tokens_seen, step + 1)
                            writer.add_scalar("training/optimizer_step", optimizer_step, step + 1)
                            writer.add_scalar("system/gpu_memory_allocated", gpu["allocated"], step + 1)
                            writer.add_scalar("system/gpu_memory_reserved", gpu["reserved"], step + 1)
                            writer.add_scalar("system/peak_vram", gpu["peak"], step + 1)
                        except Exception as e:
                            print(f"\n[WARN] TensorBoard write failed ({e})")

                running_loss = 0.0

            if (step + 1) % config.save_interval == 0:
                save_checkpoint(model, optimizer, step + 1, optimizer_step, tokens_seen,
                                best_val_loss, best_perplexity, elapsed0 + (time.time() - process_start))

            step += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — saving checkpoint ...")
        try:
            save_checkpoint(model, optimizer, step + 1, optimizer_step, tokens_seen,
                            best_val_loss, best_perplexity, elapsed0 + (time.time() - process_start))
        except KeyboardInterrupt:
            print("[INFO] Save interrupted — the last completed checkpoint is intact.")
        return

    pbar.close()
    if writer is not None:
        writer.close()
    try:
        _write_metrics_json(json_path, step, optimizer_step, tokens_seen,
                            best_val_loss, best_perplexity, elapsed0 + (time.time() - process_start))
    except Exception as e:
        print(f"[WARN] metrics.json write failed ({e})")

    if stopped:
        print("[INFO] Stopped early — checkpoint saved; will resume from here next time.")
        return
    final_path = os.path.join(config.save_dir, "final_model.pt")
    torch.save({"step": step, "optimizer_step": optimizer_step, "tokens_seen": tokens_seen,
                "model": (model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict())}, final_path)
    print("=" * 60)
    print(f"DONE — training complete. Final model saved to {final_path}")
    print("=" * 60)


# ----------------------------------------------------------------------
# INSTRUCTION FINE-TUNING (SFT)
# ----------------------------------------------------------------------
def finetune(progress_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    print(f"[INFO] Fine-tuning | device={device} | dtype={amp_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e9)

    # Load the pretrained checkpoint
    ckpt_path, _, state = find_latest_checkpoint()
    if ckpt_path is None:
        print("No pretrained checkpoint found. Run `python train.py` first.")
        return
    print(f"[INFO] Loading pretrained model from {os.path.basename(ckpt_path)} ...")
    model = Transformer3B(config)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    del state
    import gc
    gc.collect()
    model = model.to(device=device, dtype=amp_dtype)
    model.train()

    optimizer = AdamW8bit(model.parameters(), lr=config.finetune_lr,
                          weight_decay=0.01, betas=(0.9, 0.95))

    print("[INFO] Loading instruction dataset (Alpaca, no refusal training) ...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.shuffle(seed=42)

    FT_SEQ = config.finetune_seq_len
    EOS = tokenizer.eos_token_id

    def tokenize(ex):
        instruction = ex["instruction"].strip()
        inp = (ex.get("input") or "").strip()
        output = ex["output"].strip()
        prompt = "User: " + instruction + ("\n" + inp if inp else "") + "\nAssistant: "
        # Tokenize prompt and response SEPARATELY so the label boundary is exact.
        # Tokenizing them together lets BPE merge the prompt's trailing space
        # with the response's first word, masking the first output token and
        # collapsing the model to predict the space token.
        prompt_ids = tokenizer(prompt)["input_ids"]
        response_ids = tokenizer(output)["input_ids"] + [EOS]
        full_ids = (prompt_ids + response_ids)[: FT_SEQ]
        n_prompt = len(prompt_ids)
        labels = [-100] * len(full_ids)
        for i in range(len(response_ids)):
            pos = n_prompt - 1 + i
            if pos < len(full_ids) - 1:
                labels[pos] = response_ids[i]
        pad = FT_SEQ - len(full_ids)
        if pad > 0:
            full_ids = full_ids + [EOS] * pad
            labels = labels + [-100] * pad
        return {"input_ids": full_ids, "labels": labels}

    dataset = dataset.map(tokenize, remove_columns=["instruction", "input", "output"])
    dataset.set_format(type="torch", columns=["input_ids", "labels"])
    dataloader = DataLoader(dataset, batch_size=config.finetune_batch, shuffle=False)

    print(f"[INFO] Fine-tuning for {config.finetune_steps} steps ...")
    data_iter = iter(dataloader)
    running_loss = 0.0
    step = 0
    pbar = tqdm(total=config.finetune_steps, desc="Finetune", unit="step")
    try:
        while step < config.finetune_steps:
            if _stop_event.is_set():
                print("\n[INFO] Stop requested — saving ...")
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, targets=labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()
            pbar.update(1)
            if (step + 1) % 100 == 0:
                avg_ft = running_loss / 100
                pbar.set_postfix({"loss": f"{avg_ft:.4f}"})
                if progress_callback is not None:
                    progress_callback({"step": step + 1, "loss": avg_ft})
                running_loss = 0.0
            step += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — saving ...")

    pbar.close()
    out_path = os.path.join(config.save_dir, "finetuned_model.pt")
    torch.save({"model": model.state_dict()}, out_path)
    print(f"[INFO] Fine-tuned model saved to {out_path}")
    print("[INFO] Run `python train.py chat` to talk to it.")


# ----------------------------------------------------------------------
# INFERENCE
# ----------------------------------------------------------------------
def load_chat_model():
    """Load the fine-tuned model (or fall back to the pretrained checkpoint)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e9)

    ft_path = os.path.join(config.save_dir, "finetuned_model.pt")
    if os.path.isfile(ft_path):
        state = torch.load(ft_path, map_location="cpu")
    else:
        ckpt_path, _, state = find_latest_checkpoint()
        if ckpt_path is None:
            return None, None, device

    model = Transformer3B(config)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    state = None
    import gc
    gc.collect()
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    model = model.to(device=device, dtype=amp_dtype)
    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, prompt, device):
    """Generate a reply for a single user prompt."""
    prompt_text = "User: " + prompt + "\nAssistant: "
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                    max_length=config.seq_len - 128)
    input_ids = enc["input_ids"].to(device)
    output_ids = model.generate(
        input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9,
        do_sample=True, eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def chat():
    model, tokenizer, device = load_chat_model()
    if model is None:
        print("No model found. Run `python train.py` then `python train.py finetune`.")
        return
    print("[INFO] Ready. Type 'quit' or 'exit' to leave.\n")
    while True:
        try:
            prompt = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in ("quit", "exit"):
            break
        if not prompt.strip():
            continue
        reply = generate_response(model, tokenizer, prompt, device)
        print("AI:", reply)


# ----------------------------------------------------------------------
# AUTOMATIC EXPERIMENT REPORT
# ----------------------------------------------------------------------
def generate_report():
    """Write results/report.json + results/report.md from measured data."""
    os.makedirs(config.results_dir, exist_ok=True)
    csv_path = os.path.join(config.results_dir, "metrics.csv")

    rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    final_row = rows[-1] if rows else None

    best_val_loss = None
    best_perplexity = None
    if rows:
        valid = [r for r in rows if r.get("val_loss") not in (None, "")]
        if valid:
            best = min(valid, key=lambda r: float(r["val_loss"]))
            best_val_loss = float(best["val_loss"])
            bp = best.get("perplexity")
            best_perplexity = float(bp) if bp not in (None, "") else None

    ckpt_path, ckpt_step, state = find_latest_checkpoint()
    micro_step = int(final_row["micro_step"]) if final_row and final_row.get("micro_step") not in (None, "") else (ckpt_step or 0)
    optimizer_step = int(final_row["optimizer_step"]) if final_row and final_row.get("optimizer_step") not in (None, "") else 0
    tokens_seen = int(final_row["tokens_seen"]) if final_row and final_row.get("tokens_seen") not in (None, "") else (state.get("tokens_seen", 0) if state else 0)
    elapsed = float(final_row["elapsed_seconds"]) if final_row and final_row.get("elapsed_seconds") not in (None, "") else 0.0

    n_params = None
    try:
        _m = Transformer3B(config)
        n_params = sum(p.numel() for p in _m.parameters())
        _m = None
    except Exception:
        n_params = None

    report = {
        "experiment": config.experiment_name,
        "generated_at": _now_iso(),
        "model": {
            "parameters": round(n_params / 1e9, 2) if n_params else None,
            "vocab_size": config.vocab_size,
            "dim": config.dim,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "head_dim": config.dim // config.n_heads,
            "seq_len": config.seq_len,
            "tokenizer": config.tokenizer_name,
            "rope": True,
            "gated_mlp": True,
            "weight_tying": True,
        },
        "training": {
            "dataset": config.dataset_name,
            "dataset_config": config.dataset_config,
            "tokens_seen": tokens_seen,
            "micro_steps": micro_step,
            "optimizer_steps": optimizer_step,
            "batch_size": config.batch_size,
            "effective_batch_size": config.batch_size * config.grad_accum_steps,
            "learning_rate": config.learning_rate,
            "optimizer": "AdamW8bit",
            "precision": "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else ("float16" if torch.cuda.is_available() else "float32"),
            "gradient_checkpointing": config.gradient_checkpointing,
            "hardware": _hardware_info(),
            "world_size": WORLD_SIZE,
            "training_time_seconds": round(elapsed, 2),
            "tokens_per_second": round(tokens_seen / elapsed, 2) if elapsed != 0.0 else None,
        },
        "evaluation": {
            "final_val_loss": round(float(final_row["val_loss"]), 4) if final_row and final_row.get("val_loss") not in (None, "") else None,
            "final_perplexity": round(float(final_row["perplexity"]), 4) if final_row and final_row.get("perplexity") not in (None, "") else None,
            "best_val_loss": round(best_val_loss, 4) if best_val_loss is not None else None,
            "best_perplexity": round(best_perplexity, 4) if best_perplexity is not None else None,
            "benchmarks": None,
        },
        "reproducibility": {
            "seed": config.seed,
            "config": _config_dict(),
            "torch": torch.__version__,
        },
    }

    with open(os.path.join(config.results_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(config.results_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(_render_report_md(report))
    print("[INFO] Report written to results/report.json and results/report.md")


# ----------------------------------------------------------------------
# CHECKPOINT EVALUATION (mode: evaluate)
# ----------------------------------------------------------------------
def _build_val_dataloader():
    """Rebuild the held-out validation split used during training."""
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e9)

    cache_key = config.dataset_name.replace("/", "_") + (f"_{config.dataset_config}" if config.dataset_config else "")
    local_dir = os.path.join(DATA_DIR, cache_key)
    all_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))
    all_files = [f for f in all_files
                 if not any(k in os.path.basename(f).lower() for k in ("test", "val", "valid"))]
    n_val = max(1, int(round(len(all_files) * config.val_holdout)))
    val_files = all_files[-n_val:] if not (len(all_files) < n_val + 2) else all_files[:1]

    class _Blocks(IterableDataset):
        def __iter__(self):
            buf = []
            for f in val_files:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=1024):
                    for text in batch["text"].to_pylist():
                        buf.extend(tokenizer(text or " ")["input_ids"])
                        while True:
                            if len(buf) < config.seq_len + 1:
                                break
                            block = buf[: config.seq_len + 1]
                            buf = buf[config.seq_len:]
                            yield {"input_ids": torch.tensor(block[:-1], dtype=torch.long),
                                   "labels": torch.tensor(block[1:], dtype=torch.long)}

    return DataLoader(_Blocks(), batch_size=config.batch_size, shuffle=False, drop_last=True)


def _load_checkpoint_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location="cpu")
    model = Transformer3B(config)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    step = state.get("step", 0)
    optimizer_step = state.get("optimizer_step", step // config.grad_accum_steps)
    tokens_seen = state.get("tokens_seen", step * config.seq_len * config.batch_size)
    state = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    model = model.to(device=device, dtype=amp_dtype)
    model.eval()
    return model, device, {"micro_step": step, "optimizer_step": optimizer_step, "tokens_seen": tokens_seen}


def evaluate_checkpoints():
    """Evaluate every checkpoint on the SAME held-out validation set."""
    checkpoints = sorted(
        glob.glob(os.path.join(config.save_dir, "checkpoint_*.pt")),
        key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]),
    )
    if not checkpoints:
        print("No checkpoints found.")
        return

    val_loader = _build_val_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for ckpt in checkpoints:
        model, device, meta = _load_checkpoint_model(ckpt)
        val_loss = _evaluate(model, val_loader, config.validation_batches, device)
        perplexity = math.exp(val_loss) if (val_loss == val_loss and val_loss < 100.0) else float("nan")
        row = {
            "checkpoint": os.path.basename(ckpt),
            "micro_step": meta["micro_step"],
            "optimizer_step": meta["optimizer_step"],
            "tokens_seen": meta["tokens_seen"],
            "val_loss": round(val_loss, 4),
            "perplexity": round(perplexity, 4),
        }
        rows.append(row)
        print(f"{row['checkpoint']}  step={row['micro_step']}  opt={row['optimizer_step']}  tokens={row['tokens_seen']}  val_loss={row['val_loss']}  ppl={row['perplexity']}")

    os.makedirs(config.results_dir, exist_ok=True)
    csv_path = os.path.join(config.results_dir, "checkpoint_comparison.csv")
    json_path = os.path.join(config.results_dir, "checkpoint_comparison.json")
    fieldnames = ["checkpoint", "micro_step", "optimizer_step", "tokens_seen", "val_loss", "perplexity"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {csv_path} and {json_path}")


# ----------------------------------------------------------------------
# METRIC PLOTS (mode: plot)
# ----------------------------------------------------------------------
def plot_metrics():
    """Generate PNG plots from results/metrics.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(config.results_dir, "metrics.csv")
    if not os.path.isfile(path):
        print(f"No metrics at {path}")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"No metrics at {path}")
        return

    def col(key, cast=float):
        out = []
        for r in rows:
            v = r.get(key)
            if v in (None, ""):
                continue
            try:
                out.append(cast(v))
            except (ValueError, TypeError):
                continue
        return out

    micro = col("micro_step", int)

    def plot(y_key, y_label, fname):
        y = col(y_key)
        x = micro
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        if not y:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, y, linewidth=1.2)
        ax.set_title(fname.replace(".png", "").replace("_", " ").title())
        ax.set_xlabel("Micro-step")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(config.results_dir, fname)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)

    plot("train_loss", "Train loss", "training_loss.png")
    plot("val_loss", "Validation loss", "validation_loss.png")
    plot("perplexity", "Validation perplexity", "perplexity.png")
    plot("tokens_per_second", "Tokens/sec", "tokens_per_second.png")
    plot("peak_vram", "Peak VRAM (GB)", "gpu_memory.png")


# ----------------------------------------------------------------------
# INTERACTIVE CONSOLE (no-arg mode)
# ----------------------------------------------------------------------
def show_metrics():
    """Print the latest training metrics from results/metrics.csv."""
    path = os.path.join(config.results_dir, "metrics.csv")
    if not os.path.isfile(path):
        print("No metrics yet. Run 'train' first.")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No metrics yet. Run 'train' first.")
        return
    latest = rows[-1]
    for key, label in [
        ("micro_step", "micro step"),
        ("optimizer_step", "optimizer step"),
        ("tokens_seen", "tokens seen"),
        ("train_loss", "train loss"),
        ("val_loss", "val loss"),
        ("perplexity", "perplexity"),
        ("learning_rate", "learning rate"),
        ("tokens_per_second", "tokens/sec"),
        ("elapsed_seconds", "elapsed (s)"),
        ("peak_vram", "peak VRAM (GB)"),
    ]:
        print(f"  {label:<18} {latest.get(key, '')}")


def interactive_menu():
    """Run an interactive command console (used when no mode is passed)."""
    print("=" * 56)
    print("  BoredomLM-1.18B - interactive console")
    print("=" * 56)
    print("  commands:")
    print("    train      start pretraining (resumes from checkpoint)")
    print("    finetune   instruction fine-tune (SFT)")
    print("    chat       chat with the fine-tuned model")
    print("    evaluate   compare all checkpoints (validation loss)")
    print("    plot       generate PNG plots from metrics")
    print("    report     write results/report.json + report.md")
    print("    metrics    show the latest training metrics")
    print("    help       list commands")
    print("    exit       quit")
    print("=" * 56)
    while True:
        try:
            cmd = input("\ncmd: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if cmd in ("exit", "quit"):
            print("bye")
            break
        if not cmd:
            continue
        if cmd == "help":
            print("  train / finetune / chat / evaluate / plot / report / metrics / help / exit")
        elif cmd == "metrics":
            show_metrics()
        elif cmd == "train":
            train()
        elif cmd == "finetune":
            finetune()
        elif cmd == "chat":
            chat()
        elif cmd == "evaluate":
            evaluate_checkpoints()
        elif cmd == "plot":
            plot_metrics()
        elif cmd == "report":
            generate_report()
        else:
            print(f"  unknown command: {cmd} (type 'help')")


def main():
    parser = argparse.ArgumentParser(description="Single-file local LLM")
    parser.add_argument("mode", nargs="?", default=None,
                        choices=["train", "finetune", "chat", "evaluate", "plot", "report"],
                        help="mode; if omitted, an interactive menu opens")
    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "finetune":
        finetune()
    elif args.mode == "chat":
        chat()
    elif args.mode == "evaluate":
        evaluate_checkpoints()
    elif args.mode == "plot":
        plot_metrics()
    elif args.mode == "report":
        generate_report()
    else:
        interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            input("\n[ERROR] Something went wrong. Press Enter to exit ...")
        except Exception:
            pass


