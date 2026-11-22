"""The whole run, as resumable stages. Safe to start any number of times.

    python -m candid.pipeline              # run (or continue) the run named by CANDID_RUN
    python -m candid.pipeline push-log F   # upload a log file into the run folder

Progress lives in runs/<run>/state.json on the Hub (see candid/store.py). A machine
that disappears mid-stage (credit ran out, host went away) loses at most the current
generation shard, eval task or training interval; the next machine skips everything
already done. A code error is recorded as state["failed"] so the supervisor does not
keep paying for the same crash.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import traceback

from candid.config import Config

log = logging.getLogger("candid.pipeline")

STAGES = ["prompts", "generate", "assemble", "eval_base", "train", "eval_tuned", "gate", "export"]
# A stage marked done whose output has gone missing (or was skipped by older code) runs again.
STAGE_OUTPUTS = {"prompts": "prompts", "assemble": "pairs", "train": "adapter", "export": "gguf"}


def _stage_fns():
    from candid import evaluate, generate, train

    def eval_base(cfg, store):
        evaluate.run(cfg.base_model, cfg, store, "base")

    def eval_tuned(cfg, store):
        evaluate.run(train.merge(cfg, store), cfg, store, "tuned")

    def gate(cfg, store):
        base = store.read_json("eval_base.json")
        tuned = store.read_json("eval_tuned.json")
        ok, text = evaluate.gate(cfg, base, tuned)
        with open(store.local("report.md"), "w") as f:
            f.write(text)
        store.push("report.md")
        store.update_state(gate="PASS" if ok else "FAIL")
        print(text, flush=True)

    def export(cfg, store):
        # Exported either way (it is cheap) so a model can be tried on a phone; report.md says
        # whether it passed the gate.
        if store.state().get("gate") != "PASS":
            log.info("gate did not pass; exporting anyway for inspection, not for release")
        train.merge(cfg, store)
        env = {**os.environ, "CANDID_WORK_DIR": os.path.abspath(store.work_dir),
               "CANDID_GGUF_QUANTS": ",".join(cfg.gguf_quants)}
        subprocess.run(["bash", "remote/export_gguf.sh"], check=True, env=env)
        store.push("gguf", replace=True)
        store.push("merged", replace=True)

    return {
        "prompts": generate.prompts,
        "generate": generate.shards,
        "assemble": generate.assemble,
        "eval_base": eval_base,
        "train": train.run,
        "eval_tuned": eval_tuned,
        "gate": gate,
        "export": export,
    }


_DEFAULT = object()


def run(cfg: Config, store, fns=None) -> dict:
    if fns is None:
        fns = _DEFAULT
    real_fns = _stage_fns() if fns is _DEFAULT else fns
    store.ensure_repo()
    state = store.state()
    stale = [st for st, out in STAGE_OUTPUTS.items() if st in state["done"] and not store.exists(out)]
    if stale:
        log.info("redoing stages with missing outputs: %s", stale)
        state = store.update_state(done=[st for st in state["done"] if st not in stale], finished=False)
    if "assemble" in state["done"] and fns is _DEFAULT:
        try:  # cheap, and the machine's log is the only window into its data
            from candid.generate import data_report

            data_report(cfg, store)
        except Exception as e:  # noqa: BLE001
            log.warning("data report failed: %s", e)
    if state.get("finished"):
        log.info("run %s already finished", cfg.run)
        return state
    store.update_state(attempt=state.get("attempt", 0) + 1, failed=None, error=None,
                       instance=os.environ.get("CONTAINER_ID"))
    for stage in STAGES:
        if store.is_done(stage):
            continue
        t0 = time.time()
        store.update_state(stage=stage, stage_started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log.info("== %s ==", stage)
        try:
            real_fns[stage](cfg, store)
        except Exception:
            err = traceback.format_exc()
            log.error("stage %s failed:\n%s", stage, err)
            fails = store.state().get("failures", {})
            fails[stage] = fails.get(stage, 0) + 1
            store.update_state(failed=stage, error=err[-3000:], failures=fails)
            raise
        store.mark_done(stage)
        log.info("%s done in %.1f min", stage, (time.time() - t0) / 60)
    return store.update_state(finished=True, stage="finished")


def push_log(cfg: Config, store, path: str) -> None:
    if not os.path.exists(path):
        return
    attempt = store.state().get("attempt", 0)
    rel = f"logs/attempt-{attempt:03d}.log"
    os.makedirs(os.path.dirname(store.local(rel)), exist_ok=True)
    with open(path, "rb") as src, open(store.local(rel), "wb") as dst:
        dst.write(src.read()[-2_000_000:])
    store.push(rel, message=f"log attempt {attempt}")


def heartbeat(store, minutes: float = 5) -> None:
    """Repeat the state line, so any recent slice of the log shows the run is alive."""
    import threading

    def loop():
        while True:
            time.sleep(minutes * 60)
            try:
                store.announce()
            except Exception:  # noqa: BLE001, S110 - a heartbeat must never take the run down
                pass

    threading.Thread(target=loop, daemon=True).start()


def quiet_logs() -> None:
    for name in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(name).setLevel(logging.WARNING)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    quiet_logs()
    cfg = Config.from_env()
    store = cfg.store()
    if len(sys.argv) > 2 and sys.argv[1] == "push-log":
        push_log(cfg, store, sys.argv[2])
    else:
        heartbeat(store)
        run(cfg, store)
