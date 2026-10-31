"""
translate.py — native script + English translation via Google's free (keyless) endpoints.

    analyze("Mi tyala call kela pan to reply nahi dila")
    -> ("Marathi-English", "मी त्याला फोन केला पण त्याने उत्तर दिले नाही", "I called him but he did not reply")

Hindi/Marathi words are transliterated with Google Input Tools (English words kept as-is),
that mix is translated to English, and the English is translated back into Hindi/Marathi
so the native-script sentence is fully Devanagari.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import requests

from text_utils import clean_elongation
from rule_tagger import LIGHT_VERBS, MARATHI_FUNCTION_WORDS

TRAIN_CSV = Path(__file__).with_name("pos_hineng_train.csv")
SYSTEM_DICT = Path("/usr/share/dict/words")  # macOS/Linux English word list; skipped if missing
WORD_RE = re.compile(r"(?<![@#\w])[^\W\d_]+")  # words, but not inside @mentions / #hashtags
LANG_NAMES = {"hi": "Hindi", "en": "English"}
MARATHI_MARKERS = frozenset()

_session = requests.Session()


@lru_cache(maxsize=1)
def _lexicons() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """(words labelled mostly 'en' in train, every train word, system dictionary words)."""
    lids: dict[str, Counter] = defaultdict(Counter)
    if TRAIN_CSV.exists():
        with TRAIN_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for word, lid in zip(json.loads(row["words"]), json.loads(row["lid"])):
                    lids[word.lower()][lid] += 1
    dictionary = frozenset()
    if SYSTEM_DICT.exists():
        dictionary = frozenset(w.lower() for w in SYSTEM_DICT.read_text().split() if len(w) >= 3)
    else:
        try:
            from nltk.corpus import wordnet as wn
            dictionary = frozenset(w.lower() for w in wn.words() if len(w) >= 3 and w.isalpha())
        except Exception:
            pass
    return frozenset(w for w, c in lids.items() if c.most_common(1)[0][0] == "en"), frozenset(lids), dictionary


def is_english(word: str, lang: str = "hi") -> bool:
    """Train-data language label first; the system dictionary only for words train never saw."""
    labelled_en, seen, dictionary = _lexicons()
    word = clean_elongation(word)
    return word in labelled_en or (word not in seen and word in dictionary)


def _get(url: str, **params):
    r = _session.get(url, params=params, timeout=3)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=4096)
def transliterate_word(word: str, lang: str) -> str:
    try:
        data = _get("https://inputtools.google.com/request", text=word, itc=f"{lang}-t-i0-und", num=1)
        return data[1][0][1][0] if data[0] == "SUCCESS" else word
    except Exception:
        return word


def analyze(text: str) -> tuple[str, str, str]:
    """Return (language label, sentence in native script, English translation)."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    lang = "hi"
    has_english = any(is_english(w, lang) for w in words)

    native = WORD_RE.sub(lambda m: m.group() if is_english(m.group().lower(), lang)
                         else transliterate_word(m.group().lower(), lang), text)
    english, detected = google_translate(native, "en")
    if detected != "hi":
        detected = lang
    # English words have no good phonetic Devanagari ("was" -> वास्), so write the whole sentence
    # in Hindi by translating the English back.
    full_native = google_translate(english, detected, source="en")[0] if has_english else native
    label = LANG_NAMES.get(detected, "Hindi") + ("-English" if has_english else "")
    return label, full_native, english


def google_translate(text: str, target: str, source: str = "auto") -> tuple[str, str]:
    """Return (translation, detected source language)."""
    try:
        first = _get("https://clients5.google.com/translate_a/t",
                     client="dict-chrome-ex", sl=source, tl=target, q=text)[0]
        return tuple(first) if isinstance(first, list) else (first, source)  # sl=auto adds the detected language
    except Exception:
        return text, source

@lru_cache(maxsize=1)
def _spacy():
    import spacy  # pip install spacy && python -m spacy download en_core_web_sm
    return spacy.load("en_core_web_sm", disable=["ner"])


def _fix_tag(t) -> str:
    # spaCy tags question words SCONJ ("How nice is...", "when is the party"); UD says ADV.
    if t.tag_ == "WRB" and (t.i == t.sent.start or t.nbor(1).pos_ == "AUX" if t.i + 1 < len(t.doc) else True):
        return "ADV"
    # "does engineering", "have a car": do/have as the main verb, not a helper.
    if t.pos_ == "AUX" and t.lemma_ in ("do", "have") and (t.dep_ not in ("aux", "auxpass") or t.head.pos_ != "VERB"):
        return "VERB"
    return t.pos_


def english_pos(text: str) -> list[dict]:
    """UD tags for clean English (the hybrid tagger is trained on code-mixed tweets, not this)."""
    return [{"token": t.text, "tag": _fix_tag(t), "source": "spacy", "conf": None}
            for t in _spacy()(text) if not t.is_space]


if __name__ == "__main__":
    import sys

    label, native, english = analyze(" ".join(sys.argv[1:]) or "Mi tyala call kela pan to reply nahi dila")
    print(label, native, english, " ".join(f"{w['token']}/{w['tag']}" for w in english_pos(english)), sep="\n")
