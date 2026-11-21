"""Run Candid training on vast.ai with whatever credit there is, and keep it going.

    python vast/launch.py offers         # best GPUs right now, ranked by expected total job cost
    python vast/launch.py tick           # supervisor step: start / stop / launch as credit allows
    python vast/launch.py status         # credit, machine, last progress line
    python vast/launch.py logs [N]       # tail of the machine's log
    python vast/launch.py destroy <id>   # permanently delete a machine and its disk

How credit-limited training works:
  * One machine per run. Its disk holds all progress (candid/pipeline.py is resumable).
  * When the credit hits zero, vast.ai stops the machine. The disk stays (it bills a few
    cents a day for storage, so offers with cheap storage are preferred).
  * `tick` runs on a schedule. When credit is above --min-credit (default $0.10) it
    starts the stopped machine again and training continues where it stopped.
  * When the pipeline finishes or hits a code error, the machine stops itself.

Code reaches the machine without any third-party account: it is embedded (gzip+base64)
in the instance's environment at creation. If the branch exists on GitHub, the machine
uses that instead, which is how fixes reach an existing machine.

Key: VAST_API_KEY. In a Claude cloud session it is a network secret injected by the
proxy, so the value here is a placeholder.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = "vllm/vllm-openai:v0.34.0"
DISK_GB = 80
REF_TFLOPS = 82.6          # RTX 4090, the reference for job-time estimates
JOB_HOURS_REF = 6.0        # whole v0 pipeline on the reference GPU (estimate)
DOWNLOAD_GB = 30           # docker image + base model + datasets on a fresh machine
IDLE_DAYS = 2.0            # expected days stopped while waiting for top-ups (storage still bills)
STUCK_MINUTES = 25         # running this long without the pipeline ever starting = broken bootstrap
BRANCH = "claude/yeni-ai-model-design-zbit2y"
REPO = "https://github.com/deniz-ulku/BoredomLM"
CODE_PATHS = ["candid", "remote", "pyproject.toml"]

BOOTSTRAP = f"""
mkdir -p /workspace/src && cd /workspace/src && rm -rf code code-git
if git clone -q --depth 1 -b {BRANCH} {REPO} code-git 2>/dev/null && [ -f code-git/candid/pipeline.py ]; then
  mv code-git code; echo "[bootstrap] code from GitHub"
else
  mkdir code && echo "$CANDID_CODE" | base64 -d | tar xz -C code; echo "[bootstrap] embedded code"
fi
pip install -q vastai==1.11.2
cd code && bash remote/run.sh
"""


# ----- vast client -------------------------------------------------------------

def vast():
    from vastai import VastAI
    from vastai.api import client as vclient

    # vastai calls Session.send() directly, which skips requests' environment CA settings.
    ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if ca and not getattr(vclient.VastSession, "_candid_ca", False):
        base_init = vclient.VastSession.__init__

        def init(self, *a, **kw):
            base_init(self, *a, **kw)
            self.verify = ca

        vclient.VastSession.__init__ = init
        vclient.VastSession._candid_ca = True
    return VastAI(api_key=os.environ.get("VAST_API_KEY") or "proxy-injected")


def code_bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in CODE_PATHS:
            tar.add(os.path.join(ROOT, p), arcname=p,
                    filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t)
    return base64.b64encode(buf.getvalue()).decode()


# ----- offers ------------------------------------------------------------------

def expected_cost(o: dict) -> float:
    """Expected $ for the whole job on this offer: downloads + work scaled by compute + idle storage."""
    perf = o.get("total_flops") or 0
    if perf <= 0 or not o.get("dph_total"):
        return float("inf")
    setup_h = 0.1 + (DOWNLOAD_GB * 8000 / max(o.get("inet_down") or 100, 50)) / 3600
    work = o["dph_total"] * (setup_h + JOB_HOURS_REF * REF_TFLOPS / perf)
    idle = (o.get("storage_cost") or 0.15) * DISK_GB * IDLE_DAYS / 30
    return work + idle


def find_offers(max_dph: float, limit: int = 10) -> list[dict]:
    q = (f"num_gpus=1 gpu_ram>=31 compute_cap>=800 reliability>0.98 disk_space>={DISK_GB} inet_down>=300 "
         f"cuda_vers>=12.8 dph_total<={max_dph} rentable=true verified=true")
    offers = vast().search_offers(query=q, order="dph_total", limit=100, storage=DISK_GB)
    offers.sort(key=expected_cost)
    return offers[:limit]


def _fmt(o: dict) -> str:
    return (f"{o['id']:>10}  {o['gpu_name']:<16} {o['gpu_ram'] / 1024:>3.0f} GB  ${o['dph_total']:.3f}/h  "
            f"{o.get('total_flops', 0):>6.1f} TFLOPS  {o.get('inet_down', 0):>5.0f} Mbps  "
            f"disk ${o.get('storage_cost') or 0:.2f}/GB/mo  job ~${expected_cost(o):.2f}  {o.get('geolocation', '')}")


# ----- reading the machine's log -----------------------------------------------

def parse_log(text: str) -> dict:
    """What the pipeline last reported: state line, whether it is running, last exit code."""
    out = {"state": {}, "started": False, "running": False, "exit": None}
    last_start = last_exit = -1
    for n, line in enumerate((text or "").splitlines()):
        if line.startswith("[state] "):
            try:
                out["state"] = json.loads(line[len("[state] "):])
            except json.JSONDecodeError:
                pass
        elif line.startswith("[run] start"):
            last_start, out["started"] = n, True
        elif m := re.match(r"\[run\] exit (-?\d+)", line):
            last_exit, out["exit"] = n, int(m.group(1))
    out["running"] = last_start > last_exit
    if last_exit < last_start:
        out["exit"] = None
    return out


def read_log(v, iid: int, tail: int = 400) -> str | None:
    try:
        res = v.logs(iid, tail=str(tail))
        return res if isinstance(res, str) else json.dumps(res)
    except Exception as e:  # noqa: BLE001 - logs are best effort
        print(f"(could not read logs of {iid}: {e})")
        return None


# ----- supervisor --------------------------------------------------------------

def decide(credit: float, instances: list[dict], logs: dict, now: float, run: str,
           min_credit: float) -> tuple[list[tuple[str, int | None, str]], str]:
    """Pure decision step. Returns ([(action, instance_id, detail)], summary).

    Actions: start, stop, destroy, launch, label:<new label>.
    """
    base = f"candid-{run}"
    mine = [i for i in instances if (i.get("label") or "").startswith(base)]
    have_credit = credit > min_credit
    actions: list[tuple[str, int | None, str]] = []

    if not mine:
        if have_credit:
            return [("launch", None, f"credit ${credit:.2f}")], "launching a new machine"
        return [], f"waiting for credit (have ${credit:.2f}, need > ${min_credit:.2f})"

    summaries = []
    for i in mine:
        iid, label = i["id"], i.get("label")
        status = (i.get("actual_status") or "").lower()
        stopped = status in ("exited", "stopped", "offline") or (i.get("intended_status") or "") == "stopped"
        info = parse_log(logs.get(iid)) if logs.get(iid) is not None else None
        st = (info or {}).get("state", {})
        age_min = (now - (i.get("start_date") or now)) / 60

        if label == f"{base}-done":
            summaries.append(f"finished on {iid} (gate {st.get('gate', '?')}); model is on its disk")
            continue
        if label == f"{base}-failed":
            stage = st.get("failed")
            fails = (st.get("failures") or {}).get(stage, 0) if stage else 0
            if stage and fails < 2 and have_credit:
                actions += [(f"label:{base}", iid, "retry once"), ("start", iid, f"retry stage {stage}")]
                summaries.append(f"retrying failed stage {stage}")
            else:
                summaries.append(f"blocked: stage {stage or '?'} failed {fails}x on {iid}; needs a code fix")
            continue

        if stopped:
            if have_credit:
                actions.append(("start", iid, f"credit ${credit:.2f}"))
                summaries.append(f"restarting {iid} (was at {st.get('stage', '?')})")
            else:
                summaries.append(f"{iid} stopped at {st.get('stage', '?')}; waiting for credit "
                                 f"(have ${credit:.2f}, need > ${min_credit:.2f})")
            continue

        if status == "running" and info is not None:
            if info["exit"] is not None and not info["running"]:
                code = info["exit"]
                if code == 0:
                    actions.append((f"label:{base}-done", iid, "pipeline finished"))
                elif code != 124:
                    actions.append((f"label:{base}-failed", iid, f"pipeline exited {code}"))
                actions.append(("stop", iid, "pipeline ended; idle GPU"))
                summaries.append(f"{iid} pipeline ended with {code}")
                continue
            if not info["started"] and age_min > STUCK_MINUTES:
                actions.append(("destroy", iid, f"{age_min:.0f} min without the pipeline starting"))
                summaries.append(f"{iid} bootstrap broken")
                continue
        progress = st.get("progress") or st.get("stage") or status or "starting"
        summaries.append(f"{iid} {status or 'starting'}: {progress}")
    return actions, "; ".join(summaries)


def launch(run: str, max_dph: float, max_hours: float, overrides: dict[str, str]) -> dict:
    offers = find_offers(max_dph, limit=3)
    if not offers:
        raise SystemExit(f"no suitable offer at <= ${max_dph}/h right now")
    env = {"CANDID_RUN": run, "CANDID_MAX_HOURS": str(max_hours), "CANDID_CODE": code_bundle(),
           **{f"CANDID_{k.upper()}": v for k, v in overrides.items()}}
    last_err = None
    for o in offers:  # an offer can be taken by someone else between search and create
        try:
            res = vast().create_instance(id=o["id"], image=IMAGE, disk=DISK_GB, env=env, label=f"candid-{run}",
                                         onstart_cmd=BOOTSTRAP, ssh=True, direct=True)
            if isinstance(res, dict) and res.get("new_contract"):
                print(f"launched {res['new_contract']}: {_fmt(o)}")
                return res
            last_err = res
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise SystemExit(f"could not create an instance: {last_err}")


def tick(a) -> str:
    v = vast()
    credit = float(v.show_user().get("credit") or 0)
    instances = v.show_instances()
    logs = {i["id"]: read_log(v, i["id"]) for i in instances
            if (i.get("label") or "").startswith(f"candid-{a.run}")}
    actions, summary = decide(credit, instances, logs, time.time(), a.run, a.min_credit)
    for act, iid, why in actions:
        print(f"{act} {iid or ''}: {why}")
        if act == "launch":
            launch(a.run, a.max_dph, a.max_hours, dict(kv.split("=", 1) for kv in a.set or []))
        elif act == "start":
            v.start_instance(iid)
        elif act == "stop":
            v.stop_instance(iid)
        elif act == "destroy":
            v.destroy_instance(iid)
        elif act.startswith("label:"):
            v.label_instance(iid, act.split(":", 1)[1])
    line = f"[tick] credit ${credit:.2f} | {summary}"
    print(line)
    return line


# ----- commands ----------------------------------------------------------------

def _mine(v, run):
    return [i for i in v.show_instances() if (i.get("label") or "").startswith(f"candid-{run}")]


def cmd_status(a):
    v = vast()
    print(f"credit: ${float(v.show_user().get('credit') or 0):.2f}")
    for i in _mine(v, a.run):
        hours = (time.time() - (i.get("start_date") or time.time())) / 3600
        print(f"{i['id']} {i.get('label')} {i.get('actual_status')} {i.get('gpu_name')} "
              f"${i.get('dph_total', 0):.3f}/h, created {hours:.1f} h ago")
        info = parse_log(read_log(v, i["id"]) or "")
        print("  last state:", json.dumps(info["state"]), "| pipeline running:", info["running"],
              "| last exit:", info["exit"])


def cmd_logs(a):
    v = vast()
    for i in _mine(v, a.run):
        print(f"===== {i['id']} =====")
        print(read_log(v, i["id"], tail=a.n) or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("offers")
    p.add_argument("--max-dph", type=float, default=1.0)
    p.set_defaults(fn=lambda a: [print(_fmt(o)) for o in find_offers(a.max_dph)])
    p = sub.add_parser("tick")
    p.add_argument("--min-credit", type=float, default=0.10)
    p.add_argument("--max-dph", type=float, default=1.0)
    p.add_argument("--max-hours", type=float, default=12, help="time limit per start; the next tick restarts")
    p.add_argument("--set", action="append", help="config override for a NEW machine, e.g. --set n_image=50")
    p.set_defaults(fn=tick)
    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("logs")
    p.add_argument("n", type=int, nargs="?", default=80)
    p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("destroy")
    p.add_argument("id", type=int)
    p.set_defaults(fn=lambda a: print(vast().destroy_instance(a.id)))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
