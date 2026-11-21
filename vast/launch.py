"""Start, watch and stop Candid training runs on vast.ai, with a hard budget.

    python vast/launch.py offers                       # cheapest suitable GPUs right now
    python vast/launch.py launch --hub-repo you/candid-4b-v0 [--max-dph 0.6] [--max-hours 10]
    python vast/launch.py status                       # running instances, cost so far
    python vast/launch.py destroy <instance_id>

Budget protection, three independent layers:
  1. launch refuses if (spent so far + max_dph * max_hours) would exceed --budget (default $50).
  2. the remote pipeline runs under `timeout max_hours` and destroys its own instance when done.
  3. `status` destroys any instance that has run longer than its max_hours.

The API key is read from VAST_API_KEY. In a Claude cloud session the network proxy
adds the real key to requests, so the value here can be a placeholder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

LEDGER = os.path.join(os.path.dirname(__file__), "ledger.json")
IMAGE = "vllm/vllm-openai:v0.34.0"
REPO_URL = "https://github.com/deniz-ulku/BoredomLM"
GPUS = ["RTX_4090", "RTX_5090", "RTX_6000Ada", "L40S", "A100_SXM4", "A100_PCIE", "H100_SXM", "H100_PCIE"]
LABEL = "candid-train"


def _client():
    from vastai import VastAI

    return VastAI(api_key=os.environ.get("VAST_API_KEY") or "proxy-injected")


def _ledger() -> dict:
    try:
        with open(LEDGER) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"runs": []}


def _save_ledger(d: dict) -> None:
    with open(LEDGER, "w") as f:
        json.dump(d, f, indent=2)


def find_offers(max_dph: float, limit: int = 10) -> list[dict]:
    q = (f"num_gpus=1 gpu_ram>=24 reliability>0.98 disk_space>=150 inet_down>=500 "
         f"cuda_vers>=12.8 dph_total<={max_dph} gpu_name in [{','.join(GPUS)}]")
    offers = _client().search_offers(query=q, order="dph_total", limit=50, storage=150)
    # Prefer more compute per dollar: rough score = TFLOPS / $ per hour.
    offers.sort(key=lambda o: -(o.get("total_flops") or 0) / max(o.get("dph_total") or 1e9, 1e-6))
    return offers[:limit]


def cmd_offers(a):
    for o in find_offers(a.max_dph):
        print(f"{o['id']:>10}  {o['gpu_name']:<14} {o['gpu_ram'] / 1024:>4.0f} GB  ${o['dph_total']:.3f}/h  "
              f"{o.get('total_flops', 0):>6.1f} TFLOPS  rel {o.get('reliability2', o.get('reliability', 0)):.3f}  "
              f"{o.get('geolocation', '')}")


def spent(ledger: dict) -> float:
    return sum(r.get("cost", 0.0) for r in ledger["runs"])


def cmd_launch(a):
    ledger = _ledger()
    worst = a.max_dph * a.max_hours
    if spent(ledger) + worst > a.budget:
        sys.exit(f"refusing: spent ${spent(ledger):.2f} + worst case ${worst:.2f} > budget ${a.budget:.2f}")
    offers = find_offers(a.max_dph, limit=1)
    if not offers:
        sys.exit(f"no suitable offer at <= ${a.max_dph}/h right now")
    o = offers[0]
    env = {
        "CANDID_HUB_REPO": a.hub_repo,
        "CANDID_MAX_HOURS": str(a.max_hours),
        "CANDID_STAGES": a.stages,
        "CANDID_KEEP_ALIVE": "1" if a.keep_alive else "0",
    }
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        env[f"CANDID_{k.upper()}"] = v
    onstart = (f"cd /workspace && rm -rf candid && git clone --depth 1 -b {a.branch} {REPO_URL} candid && "
               f"cd candid && bash remote/run.sh")
    res = _client().create_instance(id=o["id"], image=IMAGE, disk=150, env=env, label=LABEL,
                                    onstart_cmd=onstart, ssh=True, direct=True)
    iid = res.get("new_contract") if isinstance(res, dict) else None
    print(json.dumps(res, indent=2, default=str))
    ledger["runs"].append({"instance": iid, "offer": o["id"], "gpu": o["gpu_name"], "dph": o["dph_total"],
                           "max_hours": a.max_hours, "started": time.time(), "cost": 0.0, "hub_repo": a.hub_repo})
    _save_ledger(ledger)
    print(f"launched instance {iid} on {o['gpu_name']} at ${o['dph_total']:.3f}/h, hard stop after {a.max_hours} h "
          f"(worst case ${o['dph_total'] * a.max_hours:.2f})")


def cmd_status(a):
    c = _client()
    ledger = _ledger()
    live = {i["id"]: i for i in c.show_instances() if i.get("label") == LABEL}
    for r in ledger["runs"]:
        inst = live.get(r["instance"])
        hours = (time.time() - r["started"]) / 3600
        if inst is not None:
            r["cost"] = round(hours * r["dph"], 3)
            state = inst.get("actual_status") or inst.get("cur_state")
            print(f"{r['instance']}: {state}, {hours:.2f} h, ~${r['cost']:.2f} so far, {r['gpu']}")
            if hours > r["max_hours"] + 0.25:
                print(f"  over its {r['max_hours']} h limit: destroying")
                c.destroy_instance(r["instance"])
        elif r.get("cost", 0) == 0:
            r["cost"] = round(min(hours, r["max_hours"]) * r["dph"], 3)  # finished; upper-bound estimate
    _save_ledger(ledger)
    print(f"total spent (estimate): ${spent(ledger):.2f}")
    try:
        user = c.show_user()
        print(f"vast.ai credit: ${float(user.get('credit', 0)):.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"(could not read account credit: {e})")


def cmd_destroy(a):
    print(_client().destroy_instance(a.id))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("offers")
    p.add_argument("--max-dph", type=float, default=0.8)
    p.set_defaults(fn=cmd_offers)
    p = sub.add_parser("launch")
    p.add_argument("--hub-repo", required=True, help="private HF repo for results, e.g. you/candid-4b-v0")
    p.add_argument("--max-dph", type=float, default=0.6, help="max $/hour")
    p.add_argument("--max-hours", type=float, default=10)
    p.add_argument("--budget", type=float, default=50)
    p.add_argument("--branch", default="claude/yeni-ai-model-design-zbit2y")
    p.add_argument("--stages", default="all")
    p.add_argument("--keep-alive", action="store_true")
    p.add_argument("--set", action="append", help="config override, e.g. --set n_image=2000")
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("destroy")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_destroy)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
