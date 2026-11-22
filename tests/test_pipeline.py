import json

from candid import evaluate
from candid.config import Config
from candid.evaluate import _final, _letter, _num, _relaxed
from candid.sources import _map_aya, _map_cauldron, _map_messages, _take


def test_config_env(monkeypatch):
    monkeypatch.setenv("CANDID_N_IMAGE", "12")
    monkeypatch.setenv("CANDID_GGUF_QUANTS", "Q4_K_M")
    monkeypatch.setenv("CANDID_LEARNING_RATE", "1e-5")
    monkeypatch.setenv("CANDID_RUN", "v9")
    monkeypatch.delenv("CANDID_WORK_DIR", raising=False)
    c = Config.from_env()
    assert c.n_image == 12 and c.gguf_quants == ["Q4_K_M"] and c.learning_rate == 1e-5
    assert c.run == "v9" and c.work_dir.endswith("runs/v9")


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

    _write(cfg, "tuned", _report(gsm8k=74.0, over_refusal=5.0))  # lost 6 points of math: beyond noise
    assert not evaluate.gate(cfg)[0]

    _write(cfg, "tuned", _report(over_refusal=31.0))  # did not reduce refusals
    assert not evaluate.gate(cfg)[0]

    _write(cfg, "tuned", _report(over_refusal=5.0, hard_limit_refusal=50.0))  # hard limits eroded
    assert not evaluate.gate(cfg)[0]

    _write(cfg, "tuned", _report(over_refusal=5.0, chartqa=None))  # missing benchmark is skipped, not failed
    assert evaluate.gate(cfg)[0]


def test_parquet_rows_reads_images_one_group_at_a_time(tmp_path):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from candid.sources import _map_cauldron, _take, parquet_rows

    def png(color):
        buf = io.BytesIO()
        Image.new("RGB", (2400, 1200), color).save(buf, format="PNG")
        return {"bytes": buf.getvalue(), "path": None}

    rows = [{"images": [png((i, 0, 0))], "texts": [{"user": f"q{i}", "assistant": "a"}]} for i in range(10)]
    path = tmp_path / "train-0.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    it = parquet_rows("repo", "chartqa/", ["images", "texts"], seed=1,
                      list_files=lambda: ["chartqa/train-0.parquet", "other/x.parquet", "README.md"],
                      download=lambda f: str(path))
    out = _take((r for ex in it for r in _map_cauldron(ex)), 4, "image")
    assert len(out) == 4 and len({r["prompt"] for r in out}) == 4
    img = Image.open(io.BytesIO(out[0]["image"]["bytes"]))
    assert img.format == "JPEG" and img.size == (1536, 768)


def test_train_helpers_cap_images_and_drop_long_pairs():
    from datasets import Dataset

    from candid.train import cap_image_size, drop_too_long

    class IP:
        def __init__(self):
            self.size = {"shortest_edge": 3136, "longest_edge": 99_999_999}

    class Proc:
        image_processor = IP()

        def __call__(self, text, images=None):
            n = len(text[0].split()) + (1000 if images else 0)
            return {"input_ids": [[0] * n]}

    proc = Proc()
    cap_image_size(proc, 1024 * 1024)
    assert proc.image_processor.size == {"shortest_edge": 3136, "longest_edge": 1024 * 1024}
    ds = Dataset.from_dict({"images": [[], ["img"], []], "prompt": ["a b", "c", "x " * 60],
                            "chosen": ["d", "e", "f"], "rejected": ["g", "h", "i"]})
    kept = drop_too_long(ds, proc, max_length=50)
    assert kept["prompt"] == ["a b"]


def test_data_report_counts_buckets_and_soft_refusals(tmp_path, capsys):
    from datasets import Dataset, Features, Image, Value

    from candid.generate import data_report, shard_name
    from candid.store import Store

    cfg = Config(work_dir=str(tmp_path / "runs" / "r"), run="r", shard_size=2)
    store = Store(None, "r", cfg.work_dir)
    feats = Features({"id": Value("string"), "bucket": Value("string"), "lang": Value("string"),
                      "prompt": Value("string"), "image": Image()})
    Dataset.from_dict({"id": ["a", "b", "c"], "bucket": ["over_refusal", "over_refusal", "general_en"],
                       "lang": ["en"] * 3, "prompt": ["p0", "p1", "p2"], "image": [None] * 3},
                      features=feats).save_to_disk(store.local("prompts"))
    store.write_json(shard_name(0), {"pairs": [{"idx": 0, "chosen": "Sure, here it is.", "rejected": "No."},
                                              {"idx": 1, "chosen": "I can't share that, but", "rejected": "x"}],
                                    "stats": {"over_refusal/total": 2, "over_refusal/kept": 2}})
    store.write_json(shard_name(1), {"pairs": [], "stats": {"general_en/total": 1, "general_en/no_clean_spec": 1}})
    rep = data_report(cfg, store)
    assert rep["over_refusal"] == {"total": 2, "kept": 2, "soft_refusal_in_chosen": 1}
    assert rep["general_en"]["no_clean_spec"] == 1
    out = capsys.readouterr().out
    assert "[report] over_refusal kept=2 soft_refusal_in_chosen=1 total=2" in out
    assert all(len(line) < 500 for line in out.splitlines())


def test_v1_selection_logic():
    from candid.generate import accept_edit, clean_edit, needs, pick_compliant

    assert needs("I'm sorry, but I can't help with that.") == "spec"
    assert needs("") == "spec"
    assert needs("Great question! Water boils at 100 C.") == "edit"
    assert needs("Water boils at 100 C at sea level.") == "none"

    original = "Great question! Here are the steps. " + "Step detail. " * 20
    assert accept_edit(original, "Here are the steps. " + "Step detail. " * 20, 0.6) is None
    assert accept_edit(original, "Steps.", 0.6) == "too_short"
    assert accept_edit(original, "Sure! " + "Step detail. " * 20, 0.6) == "unclean"
    assert accept_edit(original, "", 0.6) == "empty"
    assert clean_edit("<<<\nThe answer.\n>>>") == "The answer."

    assert pick_compliant(["I cannot help with that request.", "Here is how phishing emails usually look."]) \
        == "Here is how phishing emails usually look."
    assert pick_compliant(["I can't provide that."]) is None


def test_allowed_drop_is_noise_aware():
    from candid.evaluate import allowed_drop

    small = allowed_drop(90, 90, 300, 300, 1.0)
    big = allowed_drop(90, 90, 1000, 1000, 1.0)
    assert small > big > 1.0  # fewer questions, wider band
    assert round(big, 1) == 2.2
    assert allowed_drop(100, 100, 1000, 1000, 1.0) == 1.0  # no variance: the floor applies
