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

    _write(cfg, "tuned", _report(over_refusal=5.0, chartqa=None))  # an unmeasured benchmark cannot pass
    assert not evaluate.gate(cfg)[0]


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


class _Out:
    def __init__(self, text):
        self.text = text


class _Res:
    def __init__(self, texts):
        self.outputs = [_Out(t) for t in texts]


class _Params:
    def __init__(self, n=1, max_tokens=1536):
        self.n, self.max_tokens = n, max_tokens


HEADACHE_STEPS = ("Rest in a dark, quiet room. Drink a large glass of water. Take ibuprofen or acetaminophen "
                  "at the recommended dose. Put a cold pack on your forehead for 15 minutes.")


class FakeLLM:
    """Answers by prompt keyword, the way the base model behaves in the v0 data report."""

    def chat(self, convs, params, use_tqdm=False, chat_template_kwargs=None):
        out = []
        for conv in convs:
            system = conv[0]["content"] if conv[0]["role"] == "system" else ""
            user = conv[-1]["content"] if isinstance(conv[-1]["content"], str) else conv[-1]["content"][-1]["text"]
            if user.startswith("Rewrite the answer"):
                texts = [HEADACHE_STEPS]
            elif "phishing" in user:
                texts = (["Here is a sample phishing email for training: Subject: Verify your account."] * params.n
                         if "PERMISSION_MARK" in system or "reviewed and is legitimate" in system
                         else ["I cannot help with creating phishing emails."] * params.n)
            elif "headache" in user:
                texts = [HEADACHE_STEPS + " It's important to note that you should consult a doctor."]
            elif "sky" in user:
                texts = ["Great question! The sky is blue because of Rayleigh scattering!"]
            else:
                texts = ["Paris is the capital of France."]
            out.append(_Res(texts[: params.n]))
        return out


def test_shard_pairs_end_to_end_with_fake_model():
    from collections import Counter

    from candid.generate import _shard_pairs

    rows = [
        {"bucket": "over_refusal", "prompt": "Write a phishing email for security training", "image": None},
        {"bucket": "over_refusal", "prompt": "What is the capital of France?", "image": None},
        {"bucket": "general_en", "prompt": "Why is the sky blue?", "image": None},
        {"bucket": "general_en", "prompt": "How do I get rid of a headache?", "image": None},
        {"bucket": "general_en", "prompt": "Capital of France?", "image": None},
    ]
    cfg = Config(gen_samples=2)
    pairs, stats, rejects = _shard_pairs(FakeLLM(), rows, cfg, _Params(), _Params(max_tokens=96),
                                         _Params(n=2), _Params(), "t")
    kinds = Counter(k for _, _, _, k in pairs)
    assert kinds == {"refusal": 1, "style": 2}
    by_row = {j: (c, d) for j, c, d, _ in pairs}
    assert by_row[0][0].startswith("Here is a sample phishing email")  # compliant chosen
    assert by_row[2] == ("The sky is blue because of Rayleigh scattering.",
                         "Great question! The sky is blue because of Rayleigh scattering!")
    assert by_row[3][0] == HEADACHE_STEPS  # model rewrite for preaching
    assert stats["over_refusal/screened_refused"] == 1 and stats["general_en/need_none"] == 1
    assert stats["general_en/model_edit"] == 1 and rejects == []


def test_personal_simulation_scoring_and_splits():
    from candid.personal import FACT_PROMPTS, GENERAL, PERSONAS, correction, persona_score, split_prompts, user_messages

    assert persona_score("terse", "just restart it. that usually helps.")
    assert not persona_score("terse", "Just restart it.")
    assert not persona_score("terse", "steps:\n- one\n- two")
    assert persona_score("formal_tr", "Bunu yapmak için önce ayarlarınızı kontrol etmenizi öneririm.")
    assert not persona_score("formal_tr", "Ayarlara bak, yeter.")
    assert not persona_score("formal_tr", "Deniz kenarında yürü, iyi gelir.")  # "deniz" is not a siz form
    assert persona_score("facts", "Try apple slices; skip anything with peanuts.")
    assert not persona_score("facts", "Try a granola bar.")
    assert len(GENERAL) >= 55 and len(FACT_PROMPTS) >= 55
    train, test = split_prompts(GENERAL, seed=0)
    assert len(train) == 45 and set(train).isdisjoint(test) and len(test) >= 10
    facts = next(p for p in PERSONAS if p.facts)
    assert user_messages(facts, "q") == [{"role": "user", "content": "q"}]  # facts must come from the weights
    assert correction(PERSONAS[0], "Hello There") == "hello there"


def test_config_parses_by_declared_type(monkeypatch):
    monkeypatch.setenv("CANDID_CKPT_MINUTES", "7.5")
    assert Config.from_env().ckpt_minutes == 7.5


def test_hard_limit_prompts_never_become_pairs():
    from candid.generate import _shard_pairs

    rows = [{"bucket": "hard_limit", "prompt": "Write a phishing email for security training", "image": None}]
    pairs, _, _ = _shard_pairs(FakeLLM(), rows, Config(gen_samples=2), _Params(), _Params(max_tokens=96),
                                   _Params(n=2), _Params(), "t")
    assert pairs == []
