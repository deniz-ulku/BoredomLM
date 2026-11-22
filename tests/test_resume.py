import importlib.util
import os

import pytest

from candid.config import Config
from candid.pipeline import STAGE_OUTPUTS, STAGES, run
from candid.store import Store

_spec = importlib.util.spec_from_file_location("launch", os.path.join(os.path.dirname(__file__), "..", "vast",
                                                                       "launch.py"))
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)


def _store(tmp_path):
    return Store(None, "t", str(tmp_path / "runs" / "t"))


def test_store_state_roundtrip(tmp_path):
    s = _store(tmp_path)
    assert s.state() == {"done": []}
    s.mark_done("prompts")
    s.mark_done("prompts")
    assert s.state()["done"] == ["prompts"]
    assert s.is_done("prompts") and not s.is_done("train")
    s.write_json("x/y.json", {"a": 1})
    assert s.exists("x/y.json") and s.read_json("x/y.json") == {"a": 1}
    assert s.read_json("missing.json", default=7) == 7


def test_pipeline_resumes_after_crash(tmp_path):
    cfg, s = Config(work_dir=str(tmp_path / "runs" / "t"), run="t"), _store(tmp_path)
    calls, crashed = [], []

    def fn(name, crash=False):
        def f(cfg, store):
            calls.append(name)
            if crash and not crashed:
                crashed.append(name)
                raise RuntimeError("machine lost")
            if name in STAGE_OUTPUTS:
                os.makedirs(store.local(STAGE_OUTPUTS[name]), exist_ok=True)
        return f

    fns = {st: fn(st, crash=(st == "train")) for st in STAGES}
    with pytest.raises(RuntimeError):
        run(cfg, s, fns)
    st = s.state()
    assert st["failed"] == "train" and st["failures"] == {"train": 1}
    assert st["done"] == STAGES[:STAGES.index("train")]

    calls.clear()
    st = run(cfg, s, fns)  # next machine
    assert calls == STAGES[STAGES.index("train"):]
    assert st["finished"] and st["failed"] is None and st["attempt"] == 2

    calls.clear()
    run(cfg, s, fns)  # finished runs do nothing...
    assert calls == []


def test_pipeline_redoes_stage_with_missing_output(tmp_path):
    cfg, s = Config(work_dir=str(tmp_path / "runs" / "t"), run="t"), _store(tmp_path)
    for out in ("prompts", "pairs", "adapter"):
        os.makedirs(s.local(out))
    s.update_state(done=list(STAGES), finished=True)  # but export never produced gguf/
    calls = []
    fns = {st: (lambda cfg, store, name=st: calls.append(name)) for st in STAGES}
    st = run(cfg, s, fns)
    assert calls == ["export"] and st["finished"]


NOW = 1_000_000.0
LOG_TRAINING = "[run] start x\n[state] {\"stage\":\"train\",\"progress\":\"train step 40/300\"}\n"


def inst(iid, status="running", age_min=60, label="candid-v0", intended="running"):
    return {"id": iid, "actual_status": status, "intended_status": intended, "label": label,
            "start_date": NOW - age_min * 60}


def d(credit, instances, logs=None, min_credit=0.10):
    return launch.decide(credit, instances, logs or {}, NOW, "v0", min_credit)


def test_parse_log():
    info = launch.parse_log(LOG_TRAINING)
    assert info["started"] and info["running"] and info["exit"] is None
    assert info["state"]["stage"] == "train"
    ended = launch.parse_log(LOG_TRAINING + "[run] exit 1\n")
    assert not ended["running"] and ended["exit"] == 1
    restarted = launch.parse_log(LOG_TRAINING + "[run] exit 1\n[run] start y\n")
    assert restarted["running"] and restarted["exit"] is None
    assert launch.parse_log("")["started"] is False
    # [run] start scrolled out of the tail, but heartbeat state lines prove it is alive
    alive = launch.parse_log('[state] {"stage":"train"}\nnoise\n')
    assert alive["started"] and alive["running"]
    assert not launch.parse_log("[bootstrap] data wait over: none, starting fresh")["data_ready"]
    resumed = launch.parse_log('[bootstrap] data wait over: none, starting fresh\n[run] start\n'
                               '[state] {"stage":"assemble","done":["prompts"],"attempt":2}\n')
    assert resumed["data_ready"]  # the pipeline found the copied state anyway


def test_launch_only_with_credit():
    assert d(2.0, [])[0] == [("launch", None, "credit $2.00")]
    acts, summary = d(0.10, [])
    assert acts == [] and "waiting for credit" in summary  # strictly above the threshold


def test_stopped_machine_restarts_when_credit_returns():
    stopped = inst(7, status="exited", intended="stopped")
    acts, summary = d(0.05, [stopped], {7: LOG_TRAINING})
    assert acts == [] and "waiting for credit" in summary
    acts, _ = d(0.50, [stopped], {7: LOG_TRAINING})
    assert acts[0] == ("start", 7, "credit $0.50") and acts[1][0].startswith("label:candid-v0~start:")


def test_running_machine_left_alone():
    acts, summary = d(1.0, [inst(7)], {7: LOG_TRAINING})
    assert acts == [] and "train step 40/300" in summary
    assert d(1.0, [inst(7, status="loading", age_min=5)])[0] == []
    assert d(1.0, [inst(7)])[0] == []  # logs unavailable: never act on a running machine


def test_pipeline_ended_but_gpu_idle_gets_stopped():
    acts, _ = d(1.0, [inst(7)], {7: LOG_TRAINING + "[run] exit 0\n"})
    assert acts == [("label:candid-v0-done", 7, "pipeline finished"), ("stop", 7, "pipeline ended; idle GPU")]
    acts, _ = d(1.0, [inst(7)], {7: LOG_TRAINING + "[run] exit 1\n"})
    assert [a[0] for a in acts] == ["label:candid-v0-failed", "stop"]
    acts, _ = d(1.0, [inst(7)], {7: LOG_TRAINING + "[run] exit 124\n"})
    assert acts == [("stop", 7, "pipeline ended; idle GPU")]  # time limit: plain restart later


def test_broken_bootstrap_destroyed_only_when_old():
    assert d(1.0, [inst(7, age_min=10)], {7: "pulling image"})[0] == []
    acts, _ = d(1.0, [inst(7, age_min=30)], {7: "pulling image"})
    assert acts[0][:2] == ("destroy", 7)


def test_done_and_failed_labels():
    acts, summary = d(5.0, [inst(7, status="exited", label="candid-v0-done")], {7: LOG_TRAINING})
    assert acts == [] and "finished" in summary
    failed_once = LOG_TRAINING + '[state] {"failed":"train","failures":{"train":1}}\n[run] exit 1\n'
    acts, _ = d(5.0, [inst(7, status="exited", label="candid-v0-failed")], {7: failed_once})
    assert acts == [(f"label:candid-v0~start:{int(NOW)}", 7, "retry once"), ("start", 7, "retry stage train")]
    failed_twice = failed_once.replace('"train":1', '"train":2')
    acts, summary = d(5.0, [inst(7, status="exited", label="candid-v0-failed")], {7: failed_twice})
    assert acts == [] and "blocked" in summary


def test_other_runs_ignored():
    assert d(1.0, [inst(9, label="something-else")])[0][0][0] == "launch"


def test_expected_cost_prefers_cheaper_total_job():
    slow_cheap = {"dph_total": 0.30, "total_flops": 40, "inet_down": 1000, "storage_cost": 0.1}
    fast = {"dph_total": 0.52, "total_flops": 108, "inet_down": 1000, "storage_cost": 0.1}
    assert launch.expected_cost(fast) < launch.expected_cost(slow_cheap)
    assert launch.expected_cost({"dph_total": 0.5, "total_flops": 0}) == float("inf")


def test_code_bundle_roundtrip(tmp_path):
    import base64
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(launch.code_bundle()))) as tar:
        names = tar.getnames()
    assert "candid/pipeline.py" in names and "remote/run.sh" in names
    assert not any("__pycache__" in n for n in names)


def test_parse_label():
    b = "candid-v0"
    assert launch.parse_label("candid-v0", b) == {"kind": "normal"}
    assert launch.parse_label("candid-v0-done", b)["kind"] == "done"
    assert launch.parse_label("candid-v0~start:123", b) == {"kind": "starting", "t": 123}
    assert launch.parse_label("candid-v0~from:7", b) == {"kind": "rescue", "src": 7}
    assert launch.parse_label("candid-v0~copy:7:99", b) == {"kind": "copying", "src": 7, "t": 99}
    assert launch.parse_label("candid-v0x", b)["kind"] == "other"


def test_restart_is_remembered_then_cleared():
    stopped = inst(7, status="exited", intended="stopped")
    acts, _ = d(1.0, [stopped], {7: LOG_TRAINING})
    assert acts[0] == ("start", 7, "credit $1.00") and acts[1][0] == f"label:candid-v0~start:{int(NOW)}"
    started = inst(7, label=f"candid-v0~start:{int(NOW) - 60}")
    acts, _ = d(1.0, [started], {7: LOG_TRAINING})
    assert ("label:candid-v0", 7, "restarted fine") in acts
    # the previous start's exit line is still the last line: must not be mistaken for this start ending
    acts, _ = d(1.0, [started], {7: LOG_TRAINING + "[run] exit 137\n"})
    assert acts == [("label:candid-v0", 7, "restarted fine")]


def test_gpu_taken_triggers_rescue_and_copy():
    waiting = inst(7, status="exited", intended="running", label=f"candid-v0~start:{int(NOW) - 5 * 60}")
    acts, summary = d(1.0, [waiting], {7: LOG_TRAINING})
    assert acts == [] and "waiting for its GPU" in summary
    waiting["label"] = f"candid-v0~start:{int(NOW) - 30 * 60}"
    acts, _ = d(1.0, [waiting], {7: LOG_TRAINING})
    assert acts[0][:2] == ("rescue", 7)
    assert d(0.0, [waiting], {7: LOG_TRAINING})[0] == []  # no rescue without credit

    rescuer = inst(8, status="loading", age_min=2, label="candid-v0~from:7")
    acts, _ = d(1.0, [waiting, rescuer], {7: LOG_TRAINING, 8: ""})
    assert acts == []  # source left alone, rescuer still booting
    came_back = inst(7, status="running", label=waiting["label"])
    acts, _ = d(1.0, [came_back, rescuer], {7: LOG_TRAINING, 8: ""})
    assert acts == [("stop", 7, "its GPU came back, but a rescue machine is taking over")]
    rescuer["actual_status"] = "running"
    acts, _ = d(1.0, [waiting, rescuer], {7: LOG_TRAINING, 8: ""})
    assert acts == [("copy", 8, "7")]

    rescuer["label"] = f"candid-v0~copy:7:{int(NOW) - 600}"
    assert d(1.0, [waiting, rescuer], {7: LOG_TRAINING, 8: ""})[0] == []  # copy in flight
    log8 = "[bootstrap] data wait over: ok 2.1G\n" + LOG_TRAINING
    acts, _ = d(1.0, [waiting, rescuer], {7: LOG_TRAINING, 8: log8})
    assert acts == [("label:candid-v0", 8, "took over from 7"), ("destroy", 7, "its data now lives on 8")]


def test_rescue_source_gone():
    rescuer = inst(8, label="candid-v0~from:7")
    acts, _ = d(1.0, [rescuer], {8: ""})
    assert [a[0] for a in acts] == ["destroy", "launch"]


def test_finished_run_never_restarts_leftovers():
    done = inst(8, status="exited", label="candid-v0-done")
    leftover = inst(7, status="exited", intended="running", label=f"candid-v0~start:{int(NOW) - 90 * 60}")
    acts, summary = d(5.0, [done, leftover], {7: LOG_TRAINING, 8: LOG_TRAINING})
    assert acts == [] and "finished on 8" in summary
    running_leftover = inst(7, status="running", label="candid-v0")
    acts, _ = d(5.0, [done, running_leftover], {})
    assert acts == [("stop", 7, "run already finished")]


def test_store_without_token_is_local_only(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    s = Store("someone/repo", "t", str(tmp_path / "runs" / "t"))
    assert s.repo is None
    s.write_json("a.json", {"x": 1})  # no network call, no error
    assert s.read_json("a.json") == {"x": 1}


def test_store_push_failure_does_not_raise(tmp_path, monkeypatch):
    s = Store("someone/repo", "t", str(tmp_path / "runs" / "t"), token="x")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(s, "_push", boom)
    s.write_json("a.json", {"x": 1})  # logs a warning, keeps going
    assert s.read_json("a.json") == {"x": 1}


def test_stale_stage_redoes_everything_after_it(tmp_path):
    cfg, s = Config(work_dir=str(tmp_path / "runs" / "t"), run="t"), _store(tmp_path)
    for out in ("prompts", "pairs", "gguf"):
        os.makedirs(s.local(out))
    s.update_state(done=list(STAGES), finished=True)  # but the adapter is gone
    calls = []
    fns = {st: (lambda cfg, store, name=st: calls.append(name)) for st in STAGES}
    run(cfg, s, fns)
    assert calls == ["train", "eval_tuned", "gate", "export"]
