"""Run state and artifacts.

Default (no repo): local-only. The work dir sits on the vast.ai machine's disk, which
survives the machine being stopped when credit runs out, so a restart continues from it.

Optional (CANDID_HUB_REPO set): everything is also mirrored to a private Hugging Face
repo under runs/<run>/, so a run can continue on a different machine:

    <work_dir>/<rel>  <->  hub://<repo>/runs/<run>/<rel>
"""

from __future__ import annotations

import json
import logging
import os
import time

log = logging.getLogger("candid.store")

STATE = "state.json"


class Store:
    def __init__(self, repo: str | None, run: str, work_dir: str, token: str | None = None):
        self.repo, self.run, self.work_dir = repo or None, run, work_dir
        self.token = token or os.environ.get("HF_TOKEN")
        if self.repo and not self.token:
            log.warning("CANDID_HUB_REPO is set but HF_TOKEN is not: working local-only")
            self.repo = None
        self._files: set[str] | None = None
        os.makedirs(work_dir, exist_ok=True)

    # ----- paths ---------------------------------------------------------------
    def local(self, rel: str) -> str:
        return os.path.join(self.work_dir, rel)

    def remote(self, rel: str) -> str:
        return f"runs/{self.run}/{rel}".rstrip("/")

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def ensure_repo(self) -> None:
        if self.repo:
            self._api().create_repo(self.repo, private=True, exist_ok=True)

    def _remote_files(self, refresh: bool = False) -> set[str]:
        if not self.repo:
            return set()
        if self._files is None or refresh:
            try:
                self._files = set(self._api().list_repo_files(self.repo))
            except Exception as e:  # noqa: BLE001 - a new repo has no files yet
                log.warning("list_repo_files failed: %s", e)
                self._files = set()
        return self._files

    # ----- queries -------------------------------------------------------------
    def exists(self, rel: str) -> bool:
        if os.path.exists(self.local(rel)):
            return True
        r = self.remote(rel)
        return any(f == r or f.startswith(r + "/") for f in self._remote_files())

    def pull(self, rel: str) -> bool:
        """Make <rel> (file or folder) available locally. Returns False if it exists nowhere."""
        if os.path.exists(self.local(rel)):
            return True
        if not self.repo or not self.exists(rel):
            return False
        from huggingface_hub import snapshot_download

        r = self.remote(rel)
        root = os.path.dirname(self.work_dir.rstrip("/"))
        # local_dir layout mirrors the repo, so work_dir must be <root>/runs/<run>.
        if os.path.basename(root) != "runs":
            raise RuntimeError(f"work_dir must be <root>/runs/<run>, got {self.work_dir}")
        snapshot_download(self.repo, allow_patterns=[r, r + "/**"], local_dir=os.path.dirname(root),
                          token=self.token)
        return os.path.exists(self.local(rel))

    # ----- writes --------------------------------------------------------------
    def push(self, rel: str, message: str | None = None, replace: bool = False) -> None:
        """Mirror <rel> to the Hub. Best effort: the local disk stays the source of truth, so a
        failed upload is logged and retried with the next push instead of stopping the run."""
        if not self.repo:
            return
        try:
            self._push(rel, message, replace)
        except Exception as e:  # noqa: BLE001
            log.warning("upload of %s failed (will retry with the next push): %s", rel, e)

    def _push(self, rel: str, message: str | None, replace: bool) -> None:
        path = self.local(rel)
        api, msg = self._api(), message or f"{self.run}: {rel}"
        if os.path.isdir(path):
            api.upload_folder(repo_id=self.repo, folder_path=path, path_in_repo=self.remote(rel),
                              commit_message=msg, delete_patterns="*" if replace else None,
                              ignore_patterns=[".cache/**"])
        else:
            api.upload_file(path_or_fileobj=path, path_in_repo=self.remote(rel), repo_id=self.repo,
                            commit_message=msg)
        self._files = None

    def sync_up(self, rels) -> list[str]:
        """Upload outputs that exist locally but not yet on the Hub (e.g. made before mirroring)."""
        if not self.repo:
            return []
        missing = [r for r in rels if os.path.exists(self.local(r)) and not self._remote_has(r)]
        for r in missing:
            self.push(r, message=f"{self.run}: backfill {r}")
        return missing

    def _remote_has(self, rel: str) -> bool:
        r = self.remote(rel)
        return any(f == r or f.startswith(r + "/") for f in self._remote_files())

    def write_json(self, rel: str, obj, push: bool = True) -> None:
        path = self.local(rel)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        if push:
            self.push(rel)

    def read_json(self, rel: str, default=None):
        if not self.pull(rel):
            return default
        with open(self.local(rel)) as f:
            return json.load(f)

    # ----- run state -----------------------------------------------------------
    def state(self) -> dict:
        s = self.read_json(STATE, default=None) or {}
        s.setdefault("done", [])
        return s

    def update_state(self, **kw) -> dict:
        s = self.state()
        s.update(kw)
        s["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.write_json(STATE, s)
        self.announce(s)
        return s

    def announce(self, s: dict | None = None) -> None:
        """Print the state line the supervisor reads from the machine's log. Keep the format stable."""
        s = s if s is not None else self.state()
        keys = ("stage", "done", "failed", "failures", "finished", "gate", "attempt", "progress")
        brief = {k: s.get(k) for k in keys if s.get(k)}
        print("[state] " + json.dumps(brief, separators=(",", ":")), flush=True)

    def is_done(self, stage: str) -> bool:
        return stage in self.state()["done"]

    def mark_done(self, stage: str) -> None:
        s = self.state()
        if stage not in s["done"]:
            s["done"].append(stage)
        self.update_state(done=s["done"])
