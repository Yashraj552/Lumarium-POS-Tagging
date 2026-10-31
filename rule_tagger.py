"""
rule_tagger.py — Tier 1: deterministic morphology / heuristic rules.

Design change vs. the original blueprint
----------------------------------------
Rules run FIRST in the ensemble, so a wrong rule overrides a correct
transformer. Suffix heuristics like "-er -> ADJ" or "-se -> NOUN" misfire
constantly on plain English ("teacher", "house", "phone"), so here:

  * every rule is *calibrated* on train.json: we measure how often it is right
    and use that precision as its confidence instead of a blanket 1.0;
  * rules whose measured precision is below `min_precision` are switched off
    automatically, so bad heuristics can't poison the ensemble;
  * the fused-suffix rule only fires when the root is a known English noun
    ("dogne" -> "dog" + "ne") and the whole token is NOT a known word
    (so "phone" and "house" are never split).

Data contract:
    predict(token: str) -> {"token", "tag", "source": "rule", "conf"} | None
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from text_utils import normalize_token, squash_token, token_shape

logger = logging.getLogger(__name__)

HINDI_CASE_MARKERS = ("ne", "ko", "se", "ka", "ki", "ke", "me", "mein", "pe", "par")
REGIONAL_CASE_MARKERS = HINDI_CASE_MARKERS

HINDI_LIGHT_VERBS = frozenset({
    "kiya", "kiye", "kiyi", "kar", "karo", "karna", "karke", "kari", "karta", "karti", "karte",
    "karega", "karenge", "karegi", "diya", "diye", "di", "de", "dena", "dijiye", "hua", "hui",
    "hue", "hona", "liya", "liye", "gaya", "gayi", "gaye", "raha", "rahi", "rahe", "hoga", "hogi",
    "honge", "tha", "thi", "the", "chahiye", "sakta", "sakti", "sakte",
})
LIGHT_VERBS = HINDI_LIGHT_VERBS

HINDI_POSTPOSITIONS = frozenset({
    "ne", "ko", "se", "ka", "ki", "ke", "mein", "par", "pe", "tak", "liye", "saath", "bina", "paas",
})

HINDI_PRONOUNS = frozenset({
    "main", "maine", "mujhe", "mera", "meri", "mere", "hum", "humne", "humko", "hamara",
    "hamare", "hamari", "tu", "tune", "tujhe", "tera", "teri", "tere", "tum", "tumne",
    "tumko", "tumhara", "tumhare", "tumhari", "aap", "aapne", "aapko", "aapka", "aapke",
    "aapki", "yeh", "isme", "isne", "iski", "iske", "iska", "use", "usne", "uski", "uske",
    "uska", "usme", "unhe", "unhone", "unka", "unke", "unki", "inhe", "inhone", "inka",
    "inke", "inki", "kaun", "kisko", "kiska", "kiske", "kiski", "kisne", "kise", "kisi",
    "kisiko", "kuch", "apna", "apne", "apni",
})

HINDI_CONJUNCTIONS = frozenset({
    "aur", "tatha", "ya", "athva", "lekin", "kintu", "parantu", "magar", "balki", "warna",
    "kyunki", "kyonki", "jabki",
})

HINDI_PARTICLES = frozenset({
    "nahi", "nahin", "nhi", "mat", "bhi", "toh",
})
MARATHI_FUNCTION_WORDS = {}
LY_EXCEPTIONS = frozenset({
    "only", "family", "early", "reply", "apply", "supply", "fly", "july", "italy",
    "ugly", "silly", "holy", "lovely", "friendly", "lonely", "daily", "weekly",
    "monthly", "yearly", "likely", "belly", "jelly", "bully", "rally", "ally",
    "lily", "billy", "emily", "kelly", "sally", "molly", "holly", "costly", "elderly",
    "curly", "oily", "hilly", "chilly", "deadly", "lively", "timely", "comply",
})
ING_EXCEPTIONS = frozenset({"thing", "king", "ring", "sing", "wing", "bring", "sting",
                            "morning", "evening", "nothing", "something", "anything",
                            "everything", "during", "spring", "string", "swing",
                            "ceiling", "wedding", "building", "meeting", "feeling",
                            "clothing", "painting", "pudding", "darling", "ping"})
ED_EXCEPTIONS = frozenset({"bed", "red", "need", "speed", "seed", "feed", "shed", "wed",
                           "hundred", "sacred", "naked", "wicked", "breed", "weed",
                           "greed", "ted", "fred", "ahmed", "zed", "led"})
ER_EST_EXCEPTIONS = frozenset({"never", "ever", "her", "after", "water", "under", "over",
                               "other", "mother", "father", "brother", "sister", "teacher",
                               "number", "paper", "power", "order", "matter", "letter",
                               "summer", "winter", "rest", "best", "test", "west", "interest",
                               "chest", "guest", "nest", "request", "protest", "forest"})

TEMPORAL_NOUNS = frozenset({
    "subah", "subha", "shaam", "sham", "dopahar", "raat", "din", "saal", "mahina",
    "hafte", "hafta", "waqt", "samay", "tarikh", "tareekh", "sadi",
})
TEMPORAL_ADVERBS = frozenset({
    "aaj", "kal", "parso", "narso", "ab", "tab", "jab", "kab", "hamesha", "roz",
    "humesha", "fatafat", "jaldi", "pehle", "baad", "abhi", "tabhi", "kabhi",
})
HINDI_PRONOUNS_WH = frozenset({
    "kaun", "kisko", "kiska", "kiski", "kiske", "kisne", "kinhe", "kinse",
})

# Marathi compatibility alias (empty for strictly Hindi-English Hinglish)
MARATHI_FUNCTION_WORDS = {}

_HYPHEN_CASE_RE =re.compile(
    rf"^([^\W\d_]{{2,}})-({'|'.join(sorted(REGIONAL_CASE_MARKERS, key=len, reverse=True))})$")


@dataclass
class Rule:
    name: str
    fn: Callable[[str, str], str | None]  # (original, normalized) -> tag | None
    conf: float
    enabled: bool = True
    learn_tag: bool = False           # let calibration replace the tag with the data's majority tag
    learned_tag: str | None = None
    precision: float | None = None    # filled by calibrate()
    support: int = 0

    def apply(self, token: str, norm: str) -> str | None:
        tag = self.fn(token, norm)
        if tag is None:
            return None
        return self.learned_tag or tag


@dataclass
class RuleBasedTagger:
    min_precision: float = 0.85
    min_support: int = 5
    known_words: set[str] = field(default_factory=set)    # every train word (lowercased)
    english_nouns: set[str] = field(default_factory=set)  # candidate roots for fused suffixes
    rules: list[Rule] = field(init=False)

    def __post_init__(self) -> None:
        self.rules = [
            # --- shape rules: near-certain ---
            Rule("punct", lambda t, n: "PUNCT" if token_shape(t) == "<PUNCT>" else None, 0.97, learn_tag=True),
            Rule("number", lambda t, n: "NUM" if token_shape(t) == "<NUM>" else None, 0.95, learn_tag=True),
            Rule("url", lambda t, n: "X" if token_shape(t) == "<URL>" else None, 0.90, learn_tag=True),
            Rule("mention", lambda t, n: "X" if token_shape(t) == "<MENTION>" else None, 0.80, learn_tag=True),
            Rule("hashtag", lambda t, n: "X" if token_shape(t) == "<HASHTAG>" else None, 0.70, learn_tag=True),
            # --- code-mixing morphology & high-precision lexicons ---
            Rule("regional_case_hyphen", self._hyphen_case, 0.95),
            Rule("regional_case_fused", self._fused_case, 0.85),
            Rule("temporal_noun", lambda t, n: "NOUN" if n in TEMPORAL_NOUNS or squash_token(t) in TEMPORAL_NOUNS else None,
                 0.92, learn_tag=True),
            Rule("temporal_adv", lambda t, n: "ADV" if n in TEMPORAL_ADVERBS or squash_token(t) in TEMPORAL_ADVERBS else None,
                 0.92, learn_tag=True),
            Rule("pronoun_wh", lambda t, n: "PRON" if n in HINDI_PRONOUNS_WH or squash_token(t) in HINDI_PRONOUNS_WH else None,
                 0.95, learn_tag=True),
            Rule("hindi_postposition", lambda t, n: "ADP" if n in HINDI_POSTPOSITIONS else None, 0.94, learn_tag=True),
            Rule("hindi_pronoun", lambda t, n: "PRON" if n in HINDI_PRONOUNS else None, 0.94, learn_tag=True),
            Rule("hindi_conjunction", lambda t, n: "CCONJ" if n in HINDI_CONJUNCTIONS else None, 0.94, learn_tag=True),
            Rule("hindi_particle", lambda t, n: "PART" if n in HINDI_PARTICLES else None, 0.94, learn_tag=True),
            Rule("light_verb", lambda t, n: "AUX" if n in LIGHT_VERBS or squash_token(t) in LIGHT_VERBS else None,
                 0.90, learn_tag=True),
            # --- English heuristics (weak ones start disabled; calibration may enable them) ---
            Rule("english_ly", lambda t, n: "ADV" if len(n) >= 5 and n.endswith("ly")
                 and n not in LY_EXCEPTIONS else None, 0.85),
            Rule("english_ing", lambda t, n: "VERB" if len(n) >= 6 and n.endswith("ing")
                 and n not in ING_EXCEPTIONS else None, 0.75, enabled=False),
            Rule("english_ed", lambda t, n: "VERB" if len(n) >= 5 and n.endswith("ed")
                 and n not in ED_EXCEPTIONS else None, 0.75, enabled=False),
            Rule("english_er_est", lambda t, n: "ADJ" if len(n) >= 6 and n.endswith(("er", "est"))
                 and n not in ER_EST_EXCEPTIONS else None, 0.60, enabled=False),
        ]

    # ---------------- morphology rules ----------------
    @staticmethod
    def _hyphen_case(token: str, norm: str) -> str | None:
        return "NOUN" if _HYPHEN_CASE_RE.match(norm) else None

    def _fused_case(self, token: str, norm: str) -> str | None:
        if "-" in norm or norm in self.known_words or not self.english_nouns:
            return None
        for marker in sorted(REGIONAL_CASE_MARKERS, key=len, reverse=True):
            if norm.endswith(marker):
                root = norm[: -len(marker)]
                if len(root) >= 3 and root in self.english_nouns:
                    return "NOUN"
        return None

    # ---------------- public API ----------------
    def normalize_token(self, token: str) -> str:
        return normalize_token(token)

    def predict(self, token: str) -> dict | None:
        """Tag one token, or return None if no (enabled) rule is confident about it."""
        if not token:
            return None
        norm = normalize_token(token)
        for rule in self.rules:
            if not rule.enabled:
                continue
            tag = rule.apply(token, norm)
            if tag is not None:
                return {"token": token, "tag": tag, "source": "rule", "conf": round(rule.conf, 4),
                        "rule": rule.name}
        return None

    def predict_sentence(self, tokens: list[str]) -> list[dict | None]:
        return [self.predict(t) for t in tokens]

    # ---------------- calibration ----------------
    def calibrate(self, sentences: list[list[dict]]) -> dict[str, dict]:
        """
        Learn lexicons from train.json, then measure every rule's precision and
        set conf/enabled accordingly. Returns a report {rule_name: stats}.
        """
        word_tags: dict[str, Counter] = defaultdict(Counter)
        for sent in sentences:
            for item in sent:
                word_tags[normalize_token(item["token"])][item["tag"]] += 1
        self.known_words = set(word_tags)
        self.english_nouns = {w for w, c in word_tags.items()
                              if len(w) >= 3 and w.isascii() and w.isalpha()
                              and c.most_common(1)[0][0] == "NOUN" and sum(c.values()) >= 2}

        # While measuring, treat words seen only once as unknown so the fused-suffix
        # rule gets a realistic test (on train every word is "known" otherwise).
        all_known = self.known_words
        self.known_words = {w for w, c in word_tags.items() if sum(c.values()) >= 2}
        for rule in self.rules:
            rule.learned_tag = None
        fired: dict[str, Counter] = defaultdict(Counter)
        for sent in sentences:
            for item in sent:
                token, gold = item["token"], item["tag"]
                norm = normalize_token(token)
                for rule in self.rules:
                    tag = rule.fn(token, norm)
                    if tag is not None:
                        fired[rule.name][gold] += 1
                        fired[rule.name]["__predicted__" + tag] += 1

        self.known_words = all_known

        report = {}
        for rule in self.rules:
            counts = fired.get(rule.name, Counter())
            golds = Counter({k: v for k, v in counts.items() if not k.startswith("__")})
            n = sum(golds.values())
            rule.support = n
            if n < self.min_support:
                report[rule.name] = {"support": n, "status": "too rare — kept defaults",
                                     "enabled": rule.enabled, "conf": rule.conf}
                continue
            predicted = next(k[13:] for k in counts if k.startswith("__predicted__"))
            if rule.learn_tag:
                top_tag, top_n = golds.most_common(1)[0]
                if top_tag != predicted:
                    rule.learned_tag = top_tag
                correct = top_n
            else:
                correct = golds[predicted]
            precision = (correct + 1) / (n + 2)  # Laplace-smoothed, so tiny samples aren't 1.0
            rule.precision = precision
            rule.conf = precision
            rule.enabled = precision >= self.min_precision
            report[rule.name] = {"support": n, "precision": round(precision, 4),
                                 "tag": rule.learned_tag or predicted, "enabled": rule.enabled,
                                 "top_gold_tags": dict(golds.most_common(3))}
        return report

    # ---------------- persistence ----------------
    def save(self, path: str | Path) -> None:
        state = {
            "min_precision": self.min_precision,
            "min_support": self.min_support,
            "known_words": sorted(self.known_words),
            "english_nouns": sorted(self.english_nouns),
            "rules": {r.name: {"conf": r.conf, "enabled": r.enabled, "learned_tag": r.learned_tag,
                               "precision": r.precision, "support": r.support} for r in self.rules},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RuleBasedTagger":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        tagger = cls(min_precision=state["min_precision"], min_support=state["min_support"],
                     known_words=set(state["known_words"]),
                     english_nouns=set(state["english_nouns"]))
        for rule in tagger.rules:
            saved = state["rules"].get(rule.name)
            if saved:
                rule.conf, rule.enabled = saved["conf"], saved["enabled"]
                rule.learned_tag, rule.precision = saved["learned_tag"], saved["precision"]
                rule.support = saved["support"]
        return tagger


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) == 4 and sys.argv[1] == "calibrate":
        from data_prep import load_json

        tagger = RuleBasedTagger()
        for name, stats in tagger.calibrate(load_json(sys.argv[2])).items():
            print(f"{name:22s} {stats}")
        tagger.save(sys.argv[3])
        print(f"Saved calibrated rules to {sys.argv[3]}")
    else:
        demo = RuleBasedTagger(english_nouns={"dog", "cat", "phone"},
                               known_words={"phone", "house", "done"})
        for tok in ["dog-ne", "Dogne", "phone", "house", "jump", "kela", "KIYAAA", "quickly",
                    "only", "teacher", "running", "!!!", "2024", "@rahul", "https://x.com", "😂"]:
            print(f"{tok:16s} -> {demo.predict(tok)}")
        print("\nTo calibrate on real data: python rule_tagger.py calibrate data/train.json models/rules.json")
