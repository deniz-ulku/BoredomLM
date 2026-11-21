import json

from candid import evaluate
from candid.config import Config
from candid.evaluate import _final, _letter, _num, _relaxed
from candid.sources import _map_aya, _map_cauldron, _map_messages, _take


def test_config_env(monkeypatch):
    monkeypatch.setenv("CANDID_N_IMAGE", "12")
    monkeypatch.setenv("CANDID_USE_SELF_JUDGE", "false")
    monkeypatch.setenv("CANDID_GGUF_QUANTS", "Q4_K_M")
    monkeypatch.setenv("CANDID_LEARNING_RATE", "1e-5")
    c = Config.from_env()
    assert c.n_image == 12 and c.use_self_judge is False and c.gguf_quants == ["Q4_K_M"] and c.learning_rate == 1e-5


def test_mappers():
    assert list(_map_messages({"messages": [{"role": "user", "content": "hi"}]})) == [{"prompt": "hi"}]
    assert list(_map_aya({"language": "Turkish", "inputs": "selam"})) == [{"prompt": "selam", "lang": "tr"}]
    assert list(_map_aya({"language": "German", "inputs": "hallo"})) == []
    from PIL import Image

    (row,) = _map_cauldron({"images": [Image.new("RGB", (8, 8))], "texts": [{"user": "q?", "assistant": "a"}]})
    assert row["prompt"] == "q?" and row["image"]["bytes"][:2] == b"\xff\xd8"
    assert list(_map_cauldron({"images": ["a", "b"], "texts": [{"user": "q"}]})) == []


def test_take_dedups_and_filters():
    rows = iter([{"prompt": "a"}, {"prompt": "a"}, {"prompt": " "}, {"prompt": "b"}, {"prompt": "c"}])
    out = _take(rows, 5, "bucket", keep=lambda r: r["prompt"] != "c")
    assert [r["prompt"] for r in out] == ["a", "b"]
    assert out[0]["id"] == "bucket-0" and out[0]["image"] is None


def test_answer_parsing():
    assert _num(_final("so 3 + 4\nAnswer: 1,234")) == 1234
    assert _letter("Reasoning...\nAnswer: (C)") == "C"
    assert _letter("Answer: b") == "B"
    assert _relaxed("10.4", "10")
    assert not _relaxed("11", "10")
    assert _relaxed("Yes", "yes")


def _write(cfg, name, d):
    with open(cfg.path(f"eval_{name}.json"), "w") as f:
        json.dump(d, f)


def _report(**kw):
    base = {"gsm8k": 80.0, "mmlu_pro": 50.0, "global_mmlu_tr": 55.0, "chartqa": 70.0, "over_refusal": 30.0,
            "hard_limit_refusal": 100.0, "hard_limit_n": 10, "style_emoji": 10.0, "style_filler": 40.0,
            "style_preachy": 20.0, "style_avg_chars": 900, "samples": []}
    return {**base, **kw}


def test_gate(tmp_path):
    cfg = Config(work_dir=str(tmp_path))
    _write(cfg, "base", _report())
    _write(cfg, "tuned", _report(gsm8k=79.5, over_refusal=5.0, style_filler=0.0))
    ok, md = evaluate.gate(cfg)
    assert ok and "PASS" in md

    _write(cfg, "tuned", _report(gsm8k=77.0, over_refusal=5.0))  # lost 3 points of math
    assert not evaluate.gate(cfg)[0]

    _write(cfg, "tuned", _report(over_refusal=31.0))  # did not reduce refusals
    assert not evaluate.gate(cfg)[0]

    _write(cfg, "tuned", _report(over_refusal=5.0, hard_limit_refusal=50.0))  # hard limits eroded
    assert not evaluate.gate(cfg)[0]

    _write(cfg, "tuned", _report(over_refusal=5.0, chartqa=None))  # missing benchmark is skipped, not failed
    assert evaluate.gate(cfg)[0]
