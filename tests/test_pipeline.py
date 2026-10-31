"""Run with:  python -m pytest -q"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep import load_csv  # noqa: E402
from ensemble import ArbiterConfig, HybridTagger, decide, tune  # noqa: E402
from hmm_tagger import HMMTagger  # noqa: E402
from metrics import score  # noqa: E402
from rule_tagger import RuleBasedTagger  # noqa: E402
from tagset import normalize_tag  # noqa: E402
from text_utils import normalize_token, tokenize  # noqa: E402
from transformer_tagger import align_first_subword  # noqa: E402


def S(text):
    return [{"token": w.rsplit("/", 1)[0], "tag": w.rsplit("/", 1)[1]} for w in text.split()]


TOY = [
    S("maine/PRON dog/NOUN ko/ADP dekha/VERB ./PUNCT"),
    S("usne/PRON cat/NOUN ko/ADP pakdla/VERB ./PUNCT"),
    S("I/PRON saw/VERB the/DET dog/NOUN ./PUNCT"),
    S("the/DET dog/NOUN was/AUX happy/ADJ !/PUNCT"),
    S("he/PRON quickly/ADV jump/VERB kela/AUX ./PUNCT"),
    S("the/DET cat/NOUN ran/VERB slowly/ADV ./PUNCT"),
] * 3

CONTRACT = {"token", "tag", "source", "conf"}


# ---------------- text utils / tagset ----------------
def test_tokenize_keeps_fused_forms():
    assert tokenize("dog-ne ko dekha!! @rahul 12:30 😂") == ["dog-ne", "ko", "dekha", "!", "!", "@rahul", "12:30", "😂"]


def test_normalize_elongation():
    assert normalize_token("Sooooo") == "soo"
    assert normalize_token("good") == "good"


def test_tag_normalization():
    assert normalize_tag("CONJ") == "CCONJ"
    assert normalize_tag("B-NOUN") == "NOUN"
    assert normalize_tag("N_NN") == "NOUN"
    assert normalize_tag("weird") == "X"


# ---------------- rules ----------------
def test_rule_contract_and_precision_guards():
    r = RuleBasedTagger(english_nouns={"dog"}, known_words={"phone", "house"})
    out = r.predict("dog-ne")
    assert CONTRACT <= out.keys() and out["tag"] == "NOUN" and out["source"] == "rule"
    assert r.predict("dogne")["tag"] == "NOUN"
    for english in ("phone", "house", "teacher", "running", "only"):
        assert r.predict(english) is None, english
    assert r.predict("kiyaaaa")["tag"] == "AUX"


def test_rule_calibration_disables_bad_rules():
    # "-ly" words that are mostly ADJ in this data -> english_ly must switch off.
    data = [S(f"{w}/ADJ") for w in ["costly", "comely", "manly", "godly", "surly", "burly", "curly"]] * 2
    r = RuleBasedTagger()
    report = r.calibrate(data)
    assert report["english_ly"]["enabled"] is False
    assert r.predict("comely") is None


def test_rule_save_load(tmp_path):
    r = RuleBasedTagger()
    r.calibrate(TOY)
    r.save(tmp_path / "r.json")
    r2 = RuleBasedTagger.load(tmp_path / "r.json")
    assert r2.predict("!") == r.predict("!")


# ---------------- HMM ----------------
def brute_force(model, tokens):
    em, _ = model._lattice(tokens)
    best, best_path = -np.inf, None
    for path in itertools.product(range(len(model.tags)), repeat=len(tokens)):
        s = model.log_start[path[0]] + em[0, path[0]] + model.log_end[path[-1]]
        s += sum(model.log_trans[a, b] + em[i + 1, b] for i, (a, b) in enumerate(zip(path, path[1:])))
        if s > best:
            best, best_path = s, list(path)
    return best_path


def test_viterbi_matches_brute_force():
    m = HMMTagger().fit(TOY)
    for sent in (["the", "dog", "ran"], ["maine", "cat", "ko", "pakadla"], ["xyz", "ko"]):
        assert m.viterbi(m._lattice(sent)[0]) == brute_force(m, sent)


def test_hmm_contract_posteriors_and_oov():
    m = HMMTagger().fit(TOY)
    out = m.predict(["the", "zebra", "ran", "!"])
    assert len(out) == 4 and all(CONTRACT <= o.keys() and o["source"] == "hmm" for o in out)
    assert all(0.0 <= o["conf"] <= 1.0 for o in out)
    assert out[1]["oov"] and out[1]["tag"] == "NOUN"  # DET _ VERB context
    post = m.posteriors(m._lattice(["the", "dog"])[0])
    assert np.allclose(post.sum(1), 1.0)


def test_hmm_save_load(tmp_path):
    m = HMMTagger().fit(TOY)
    m.save(tmp_path / "h.npz")
    m2 = HMMTagger.load(tmp_path / "h.npz")
    s = ["the", "cat", "pakadla", "!"]
    assert m.predict(s) == m2.predict(s)


# ---------------- transformer alignment ----------------
def test_first_subword_alignment():
    # words: ["pakadla", "dog-ne"] -> [CLS] paka ##dla dog - ne [SEP]
    word_ids = [None, 0, 0, 1, 1, 1, None]
    probs = np.array([[1, 0, 0], [.1, .8, .1], [.9, .05, .05], [.2, .2, .6],
                      [.9, .05, .05], [.9, .05, .05], [1, 0, 0]])
    assert align_first_subword(word_ids, probs, 2) == [(1, .8), (2, .6)]
    assert align_first_subword([None, 0, None], probs[:3], 2)[1] is None  # word with no sub-words


# ---------------- ensemble ----------------
def P(tag, src, conf, **kw):
    return {"token": "w", "tag": tag, "source": src, "conf": conf, **kw}


def test_decide_waterfall():
    cfg = ArbiterConfig(threshold=0.85, fallback="hmm", rule_overridable=True)
    assert decide(P("NOUN", "rule", .95), P("VERB", "transformer", .9), None, cfg)["source"] == "rule"
    assert decide(P("NOUN", "rule", .8), P("VERB", "transformer", .99), None, cfg)["source"] == "transformer"
    assert decide(None, P("VERB", "transformer", .9), P("NOUN", "hmm", .9), cfg)["source"] == "transformer"
    assert decide(None, P("VERB", "transformer", .6), P("NOUN", "hmm", .5), cfg)["source"] == "hmm"
    assert decide(None, P("VERB", "transformer", .6), P("VERB", "hmm", .5), cfg)["source"] == "transformer+hmm"
    cfg.fallback = "max_conf"
    assert decide(None, P("VERB", "transformer", .6), P("NOUN", "hmm", .5), cfg)["source"] == "transformer"


class FakeTransformer:
    def predict(self, tokens):
        return [{"token": t, "tag": "NOUN", "source": "transformer", "conf": 0.7} for t in tokens]


def test_hybrid_end_to_end_and_tuning():
    hybrid = HybridTagger(RuleBasedTagger(), FakeTransformer(), HMMTagger().fit(TOY))
    out = hybrid.predict(["the", "dog-ne", "ran", "!"])
    assert [o["source"] for o in out][1] == "rule" and len(out) == 4
    assert all("candidates" in o and "reason" in o for o in out)
    # tuning on toy "dev" data returns a config and a sensible accuracy
    dev = TOY[:4]
    toks = [[x["token"] for x in s] for s in dev]
    cached = [(*hybrid.candidates(t), [x["tag"] for x in s]) for t, s in zip(toks, dev)]
    cfg, acc, _ = tune(cached)
    assert 0.0 <= acc <= 1.0 and isinstance(cfg, ArbiterConfig)


def test_hybrid_without_transformer():
    hybrid = HybridTagger(RuleBasedTagger(), None, HMMTagger().fit(TOY))
    assert len(hybrid.predict(["the", "cat"])) == 2


# ---------------- data prep & metrics ----------------
def test_csv_layouts(tmp_path):
    a = tmp_path / "a.csv"
    pd.DataFrame({"Sentence #": ["1", "", "2"], "Word": ["hi", "yaar", "ok"], "POS": ["INTJ", "NOUN", "INTJ"]}).to_csv(a, index=False)
    assert [len(s) for s in load_csv(a)] == [2, 1]
    b = tmp_path / "b.csv"
    b.write_text("word,tag\nhi,INTJ\nyaar,NOUN\n,\nok,INTJ\n")
    assert [len(s) for s in load_csv(b)] == [2, 1]
    c = tmp_path / "c.csv"
    pd.DataFrame({"tokens": ["['hi', 'yaar']", "['ok', 'bye']"], "tags": ["['INTJ','NOUN']", "['INTJ','CONJ']"]}).to_csv(c, index=False)
    sents = load_csv(c)
    assert len(sents) == 2 and sents[1][1]["tag"] == "CCONJ"
    # NumPy-style lists from Kaggle exports, plus an unlabelled row that must be skipped
    d = tmp_path / "d.csv"
    pd.DataFrame({"idx": [0, 1], "words": ["['hi' \"don't\" 'go']", "['a' 'b']"],
                  "pos": ["['INTJ' 'AUX' 'VERB']", "['' '']"]}).to_csv(d, index=False)
    sents = load_csv(d)
    assert len(sents) == 1 and [x["token"] for x in sents[0]] == ["hi", "don't", "go"]


def test_metrics():
    s = score(["A", "A", "B"], ["A", "B", "B"])
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["per_tag"]["A"]["recall"] == pytest.approx(0.5)
