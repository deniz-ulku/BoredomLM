"""Hugging Face Hub helpers used on the GPU machine: progress heartbeat and result upload.

The run's results live in a private model repo (CANDID_HUB_REPO). HF_TOKEN must be
set in the vast.ai account's environment variables; it never passes through this repo.
"""

from __future__ import annotations

import json
import os
import sys
import time


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def _repo() -> str | None:
    return os.environ.get("CANDID_HUB_REPO") or None


def status(stage: str, detail: str = "") -> None:
    rec = {"stage": stage, "detail": detail, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "instance": os.environ.get("CONTAINER_ID")}
    work = os.environ.get("CANDID_WORK_DIR", "work")
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "status.json"), "w") as f:
        json.dump(rec, f)
    print(f"[status] {rec}", flush=True)
    repo = _repo()
    if not repo:
        return
    try:
        api = _api()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=os.path.join(work, "status.json"), path_in_repo="run/status.json",
                        repo_id=repo, commit_message=f"status: {stage}")
    except Exception as e:  # noqa: BLE001 - a failed heartbeat must not stop training
        print(f"[status] upload failed: {e}", flush=True)


def upload(kind: str) -> None:
    repo = _repo()
    if not repo:
        print("[upload] CANDID_HUB_REPO not set, skipping", flush=True)
        return
    work = os.environ.get("CANDID_WORK_DIR", "work")
    api = _api()
    api.create_repo(repo, private=True, exist_ok=True)
    if kind == "reports":
        api.upload_folder(folder_path=work, repo_id=repo, path_in_repo="run",
                          allow_patterns=["*.json", "*.jsonl", "*.md", "*.log"], commit_message="run reports")
    elif kind == "adapter":
        api.upload_folder(folder_path=os.path.join(work, "adapter"), repo_id=repo, path_in_repo="adapter",
                          commit_message="LoRA adapter")
    elif kind == "merged":
        api.upload_folder(folder_path=os.path.join(work, "merged"), repo_id=repo, path_in_repo="merged",
                          commit_message="merged bf16 weights")
    elif kind == "gguf":
        api.upload_folder(folder_path=os.path.join(work, "gguf"), repo_id=repo, path_in_repo="gguf",
                          allow_patterns=["*.gguf"], commit_message="GGUF for llama.cpp (iOS / Android)")
    else:
        raise SystemExit(f"unknown upload kind: {kind}")
    print(f"[upload] {kind} -> {repo}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "status":
        status(sys.argv[2], " ".join(sys.argv[3:]))
    elif cmd == "upload":
        upload(sys.argv[2])
    else:
        raise SystemExit(f"unknown command: {cmd}")
