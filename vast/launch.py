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
RESCUE_MINUTES = 20        # a stopped machine that cannot start this long after we asked gets rescued
BRANCH = "claude/yeni-ai-model-design-zbit2y"
REPO = "https://github.com/deniz-ulku/BoredomLM"
CODE_PATHS = ["candid", "remote", "pyproject.toml"]

# A rescue machine (CANDID_RESCUE=1) first waits until the previous machine's work dir
# has been copied in by the supervisor and has stopped changing (or WAIT_FOR_DATA is removed).
BOOTSTRAP = f"""
mkdir -p /workspace
if [ "${{CANDID_RESCUE:-0}}" = "1" ] && [ ! -f /workspace/.rescued ]; then
  touch /workspace/WAIT_FOR_DATA; echo "[bootstrap] waiting for data from the previous machine"
  prev=""; stable=0
  for n in $(seq 1 300); do
    [ -f /workspace/WAIT_FOR_DATA ] || break
    if [ -f /workspace/work/runs/$CANDID_RUN/state.json ]; then
      s=$(du -sb /workspace/work | cut -f1); [ "$s" = "$prev" ] && stable=$((stable+1)) || stable=0; prev=$s
      [ $stable -ge 3 ] && break
    fi
    sleep 30
  done
  touch /workspace/.rescued
  if [ -f /workspace/work/runs/$CANDID_RUN/state.json ]; then echo "[bootstrap] data wait over: ok $(du -sh /workspace/work | cut -f1)"
  else echo "[bootstrap] data wait over: none, starting fresh"; fi
fi
mkdir -p /workspace/src && cd /workspace/src && rm -rf code code-git
if git clone -q --depth 1 -b {BRANCH} {REPO} code-git 2>/dev/null && [ -f code-git/candid/pipeline.py ]; then
  mv code-git code; echo "[bootstrap] code from GitHub"
else
  python3 -c "import base64,io,os,tarfile; tarfile.open(fileobj=io.BytesIO(base64.b64decode(os.environ['CANDID_CODE']))).extractall('code')"
  echo "[bootstrap] embedded code"
fi
cd code && bash remote/run.sh
"""


def current_run() -> str:
    """The run named in remote/RUN; machines read the same file from GitHub."""
    try:
        with open(os.path.join(ROOT, "remote", "RUN")) as f:
            return (f.readline().strip() or "v1")
    except FileNotFoundError:
        return "v1"


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


def _minify(src: str) -> str:
    """Drop docstrings and comments (vast.ai caps an instance's env at 32 KB in total)."""
    import ast

    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree) + "\n"


def code_bundle() -> str:
    """The code as base64(tar.xz), small enough for an instance env var."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=9) as tar:
        for top in CODE_PATHS:
            full = os.path.join(ROOT, top)
            files = [full] if os.path.isfile(full) else sorted(
                os.path.join(d, f) for d, _, fs in os.walk(full) for f in fs if "__pycache__" not in d)
            for path in files:
                if path.endswith(".pyc"):
                    continue
                with open(path, "rb") as f:
                    data = f.read()
                if path.endswith(".py"):
                    data = _minify(data.decode()).encode()
                info = tarfile.TarInfo(os.path.relpath(path, ROOT))
                info.size, info.mode = len(data), 0o755 if path.endswith(".sh") else 0o644
                tar.addfile(info, io.BytesIO(data))
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
    out = {"state": {}, "started": False, "running": False, "exit": None, "data_ready": False}
    last_start = last_exit = -1
    for n, line in enumerate((text or "").splitlines()):
        if line.startswith("[bootstrap] data wait over: ok"):
            out["data_ready"] = True
        elif line.startswith("[state] "):
            # The pipeline repeats this line every few minutes, so it also proves the run is alive
            # when "[run] start" has already scrolled out of the log tail.
            last_start, out["started"] = max(last_start, n), True
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
    # A pipeline that resumed an earlier attempt can only have found that state in copied data.
    if out["started"] and (out["state"].get("attempt") or 0) > 1:
        out["data_ready"] = True
    return out


def read_log(v, iid: int, tail: int = 3000) -> str | None:
    try:
        res = v.logs(iid, tail=str(tail))
        return res if isinstance(res, str) else json.dumps(res)
    except Exception as e:  # noqa: BLE001 - logs are best effort
        print(f"(could not read logs of {iid}: {e})")
        return None


# ----- supervisor --------------------------------------------------------------

def parse_label(label: str, base: str) -> dict:
    """Labels carry the supervisor's memory between ticks (the tick itself keeps no state).

    <base>                  normal
    <base>-done / -failed   pipeline finished / hit a code error
    <base>~start:<t>        start requested at time t (GPU may be taken by someone else)
    <base>~from:<id>        rescue machine waiting for <id>'s data
    <base>~copy:<id>:<t>    rescue machine; copy from <id> requested at time t
    """
    rest = (label or "")[len(base):]
    if rest in ("", "-done", "-failed"):
        return {"kind": {"": "normal", "-done": "done", "-failed": "failed"}[rest]}
    if m := re.fullmatch(r"~start:(\d+)", rest):
        return {"kind": "starting", "t": int(m.group(1))}
    if m := re.fullmatch(r"~from:(\d+)", rest):
        return {"kind": "rescue", "src": int(m.group(1))}
    if m := re.fullmatch(r"~copy:(\d+):(\d+)", rest):
        return {"kind": "copying", "src": int(m.group(1)), "t": int(m.group(2))}
    return {"kind": "other"}


def decide(credit: float, instances: list[dict], logs: dict, now: float, run: str,
           min_credit: float) -> tuple[list[tuple[str, int | None, str]], str]:
    """Pure decision step. Returns ([(action, instance_id, detail)], summary).

    Actions: start, stop, destroy, launch, rescue (launch a machine to receive <id>'s data),
    copy (have vast.ai copy the source given in detail into <id>), label:<new label>.
    """
    base = f"candid-{run}"
    mine = [i for i in instances if parse_label(i.get("label"), base)["kind"] != "other"
            and (i.get("label") or "").startswith(base)]
    by_id = {i["id"]: i for i in mine}
    rescuers = {parse_label(i.get("label"), base).get("src") for i in mine} - {None}
    have_credit = credit > min_credit
    actions: list[tuple[str, int | None, str]] = []

    if not mine:
        if have_credit:
            return [("launch", None, f"credit ${credit:.2f}")], "launching a new machine"
        return [], f"waiting for credit (have ${credit:.2f}, need > ${min_credit:.2f})"

    if any(parse_label(i.get("label"), base)["kind"] == "done" for i in mine):
        # The run is finished. Other machines of this run are leftovers: never start or rescue them.
        done = [i["id"] for i in mine if parse_label(i.get("label"), base)["kind"] == "done"]
        stray = [i for i in mine if i["id"] not in done]
        for i in stray:
            if (i.get("actual_status") or "").lower() == "running":
                actions.append(("stop", i["id"], "run already finished"))
        return actions, f"finished on {done[0]}; model is on its disk" + (
            f" ({len(stray)} leftover machine(s) left untouched)" if stray else "")

    summaries = []
    for i in mine:
        iid, label = i["id"], i.get("label")
        lab = parse_label(label, base)
        status = (i.get("actual_status") or "").lower()
        stopped = status in ("exited", "stopped", "offline", "scheduling", "") or \
            (i.get("intended_status") or "") == "stopped"
        info = parse_log(logs.get(iid)) if logs.get(iid) is not None else None
        st = (info or {}).get("state", {})
        age_min = (now - (i.get("start_date") or now)) / 60

        if lab["kind"] in ("rescue", "copying"):
            src = lab["src"]
            if info is not None and info["data_ready"]:
                actions.append((f"label:{base}", iid, f"took over from {src}"))
                if src in by_id:
                    actions.append(("destroy", src, f"its data now lives on {iid}"))
                summaries.append(f"{iid} took over from {src}")
            elif status != "running":
                summaries.append(f"rescue machine {iid} starting")
            elif src not in by_id:
                # Nothing will ever arrive; a fresh machine is cheaper than waiting out the bootstrap.
                actions.append(("destroy", iid, f"source {src} is gone; nothing to copy"))
                if have_credit:
                    actions.append(("launch", None, "start the run over"))
                summaries.append(f"source machine {src} vanished; starting the run over")
            elif lab["kind"] == "rescue" or now - lab["t"] > 90 * 60:
                actions.append(("copy", iid, str(src)))
                summaries.append(f"copying {src} -> {iid}")
            else:
                summaries.append(f"copy {src} -> {iid} in progress")
            continue

        if iid in rescuers:  # data source of a rescue in progress: keep it stopped, do not touch its disk
            if not stopped:
                actions.append(("stop", iid, "its GPU came back, but a rescue machine is taking over"))
            summaries.append(f"{iid} is the data source of a rescue")
            continue

        if lab["kind"] == "done":
            summaries.append(f"finished on {iid} (gate {st.get('gate', '?')}); model is on its disk")
            continue
        if lab["kind"] == "failed":
            stage = st.get("failed")
            fails = (st.get("failures") or {}).get(stage, 0) if stage else 0
            if stage and fails < 2 and have_credit:
                # Labelled as "start requested", so a taken GPU leads to a rescue like any restart.
                actions += [(f"label:{base}~start:{int(now)}", iid, "retry once"),
                            ("start", iid, f"retry stage {stage}")]
                summaries.append(f"retrying failed stage {stage}")
            else:
                summaries.append(f"blocked: stage {stage or '?'} failed {fails}x on {iid}; needs a code fix")
            continue

        if stopped:
            waited = (now - lab["t"]) / 60 if lab["kind"] == "starting" else 0
            if have_credit and waited > RESCUE_MINUTES:
                actions.append(("rescue", iid, f"could not start for {waited:.0f} min (GPU taken?)"))
                summaries.append(f"{iid} cannot restart; moving its data to a new machine")
            elif have_credit and lab["kind"] == "starting":
                summaries.append(f"{iid} start requested {waited:.0f} min ago; waiting for its GPU")
            elif have_credit:
                actions.append(("start", iid, f"credit ${credit:.2f}"))
                actions.append((f"label:{base}~start:{int(now)}", iid, "remember when we asked"))
                summaries.append(f"restarting {iid} (was at {st.get('stage', '?')})")
            else:
                summaries.append(f"{iid} stopped at {st.get('stage', '?')}; waiting for credit "
                                 f"(have ${credit:.2f}, need > ${min_credit:.2f})")
            continue

        if status == "running" and lab["kind"] == "starting":
            # Just restarted: the log still ends with the previous start's exit line, and this
            # start may not have written its own lines yet. Judge it on the next tick.
            actions.append((f"label:{base}", iid, "restarted fine"))
            summaries.append(f"{iid} restarted")
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


def launch(run: str, max_dph: float, max_hours: float, overrides: dict[str, str], label: str | None = None,
           rescue: bool = False) -> int:
    offers = find_offers(max_dph, limit=3)
    if not offers:
        raise SystemExit(f"no suitable offer at <= ${max_dph}/h right now")
    env = {"CANDID_RUN": run, "CANDID_MAX_HOURS": str(max_hours), "CANDID_CODE": code_bundle(),
           **({"CANDID_RESCUE": "1"} if rescue else {}),
           **{f"CANDID_{k.upper()}": v for k, v in overrides.items()}}
    last_err = None
    for o in offers:  # an offer can be taken by someone else between search and create
        try:
            res = vast().create_instance(id=o["id"], image=IMAGE, disk=DISK_GB, env=env,
                                         label=label or f"candid-{run}", onstart_cmd=BOOTSTRAP, ssh=True, direct=True)
            if isinstance(res, dict) and res.get("new_contract"):
                print(f"launched {res['new_contract']}: {_fmt(o)}")
                return int(res["new_contract"])
            last_err = res
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise SystemExit(f"could not create an instance: {last_err}")


def start_copy(v, run: str, src: int, dst: int) -> None:
    """Copy src's work dir into dst (vast.ai copies host to host; src may be stopped)."""
    print(v.copy(f"C.{src}:/workspace/work/", f"C.{dst}:/workspace/work/"))
    v.label_instance(dst, f"candid-{run}~copy:{src}:{int(time.time())}")


def wait_running(v, iid: int, minutes: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        i = next((i for i in v.show_instances() if i["id"] == iid), None)
        if i and i.get("actual_status") == "running":
            return True
        time.sleep(20)
    return False


def tick(a) -> str:
    v = vast()
    credit = float(v.show_user().get("credit") or 0)
    instances = v.show_instances()
    logs = {i["id"]: read_log(v, i["id"]) for i in instances
            if (i.get("label") or "").startswith(f"candid-{a.run}")}
    actions, summary = decide(credit, instances, logs, time.time(), a.run, a.min_credit)
    overrides = dict(kv.split("=", 1) for kv in a.set or [])
    for act, iid, why in actions:
        print(f"{act} {iid or ''}: {why}")
        if act == "launch":
            launch(a.run, a.max_dph, a.max_hours, overrides)
        elif act == "rescue":
            v.stop_instance(iid)  # cancel the queued start so the old machine cannot run alongside
            new = launch(a.run, a.max_dph, a.max_hours, overrides, label=f"candid-{a.run}~from:{iid}", rescue=True)
            if wait_running(v, new, 8):  # copy right away; otherwise the next tick does it
                start_copy(v, a.run, iid, new)
        elif act == "copy":
            start_copy(v, a.run, int(why), iid)
        elif act == "start":
            print(f"  {v.start_instance(iid)}")  # e.g. resources_unavailable: GPU taken, start queued
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
    ap.add_argument("--run", default=current_run(), help="default: the name in remote/RUN")
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
