"""
data_prep.py — Member 1: turn the Kaggle CSVs into train/dev/test JSON.

Output format (every other module reads this):
    [
      [{"token": "Maine", "tag": "PRON"}, {"token": "dog-ne", "tag": "NOUN"}, ...],
      ...                                   # one inner list per sentence
    ]

The Kaggle file layout isn't guaranteed, so three layouts are auto-detected:
  A) one token per row + a sentence-id column
  B) one token per row, sentences separated by blank rows
  C) one sentence per row, tokens/tags as space-separated strings or list literals

Usage:
    python data_prep.py --train pos_hineng_train.csv --test pos_hineng_test.csv
    # optional overrides if auto-detection picks the wrong columns:
    python data_prep.py ... --token-col words --tag-col pos --sent-col sent_id
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from tagset import normalize_tag

logger = logging.getLogger(__name__)

Sentence = list[dict[str, str]]

TOKEN_CANDIDATES = ("token", "tokens", "word", "words", "text", "form", "sentence")
TAG_CANDIDATES = ("tag", "tags", "pos", "pos_tag", "pos_tags", "upos", "label", "labels")
SENT_CANDIDATES = ("sentence_id", "sent_id", "sentence_idx", "sentence #",
                   "sentence", "sid", "id", "tweet_id")


def _pick(columns: list[str], candidates: tuple[str, ...], override: str | None,
          exclude: set[str] = frozenset()) -> str | None:
    if override:
        if override not in columns:
            raise ValueError(f"Column {override!r} not in CSV. Columns: {columns}")
        return override
    lowered = {c.lower().strip(): c for c in columns if c not in exclude}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


_QUOTED_RE = re.compile(r"'((?:[^'\\]|\\.)*)'" + '|' + r'"((?:[^"\\]|\\.)*)"')


def _as_list(cell) -> list[str]:
    """
    Parse a cell that holds a whole sentence. Handles:
      "['a', 'b']"   Python/JSON list
      "['a' 'b']"    NumPy-style list (no commas) — common in Kaggle exports of
                     Hugging Face datasets; literal_eval would silently glue these
                     into ONE string, so we extract the quoted items instead
      "a b"          plain space-separated text
    """
    if isinstance(cell, list):
        return [str(x) for x in cell]
    text = str(cell).strip()
    if text.startswith("[") and text.endswith("]"):
        items = [a if a or not b else b for a, b in _QUOTED_RE.findall(text)]
        if items:
            try:  # un-escape things like \' properly
                return [ast.literal_eval(f"'{x}'") if "\\" in x else x for x in items]
            except (ValueError, SyntaxError):
                return items
        inner = text[1:-1].strip()
        return inner.replace(",", " ").split() if inner else []
    return text.split()


def load_csv(path: str | Path, token_col: str | None = None, tag_col: str | None = None,
             sent_col: str | None = None) -> list[Sentence]:
    """Read one Kaggle CSV and return a list of sentences (tags normalized to UD)."""
    df = pd.read_csv(path, skip_blank_lines=False, dtype=str, keep_default_na=False)
    cols = list(df.columns)
    tok = _pick(cols, TOKEN_CANDIDATES, token_col)
    tag = _pick(cols, TAG_CANDIDATES, tag_col)
    if tok is None or tag is None:
        raise ValueError(f"Could not find token/tag columns in {cols}. "
                         "Pass --token-col and --tag-col.")
    sid = _pick(cols, SENT_CANDIDATES, sent_col, exclude={tok, tag})
    if sid and not sent_col:
        non_blank = df[sid].str.strip().replace("", pd.NA).dropna()
        if non_blank.nunique() > 0.9 * len(df):  # unique per row => it's a token id, not a sentence id
            logger.info("Column %r looks like a per-token id; ignoring it", sid)
            sid = None

    # Layout C: whole sentence per row?
    sample = df[tok].head(20)
    if sample.map(lambda s: len(_as_list(s)) > 1).mean() > 0.5:
        logger.info("%s: detected layout C (sentence per row)", path)
        sentences = []
        for i, row in df.iterrows():
            tokens, tags = _as_list(row[tok]), _as_list(row[tag])
            if not tokens:
                continue
            if not tags or all(not g.strip() for g in tags):
                continue  # unlabelled sentence (e.g. LinCE's hidden-label test split)
            if len(tokens) != len(tags):
                logger.warning("Row %d: %d tokens vs %d tags — skipped", i, len(tokens), len(tags))
                continue
            sentences.append([{"token": t, "tag": normalize_tag(g)} for t, g in zip(tokens, tags)])
        return sentences

    sentences: list[Sentence] = []
    current: Sentence = []
    prev_id = None
    for _, row in df.iterrows():
        token, label = str(row[tok]).strip(), str(row[tag]).strip()
        if sid is not None:  # Layout A. Kaggle often fills the id only on a sentence's first row.
            cur_id = str(row[sid]).strip()
            if cur_id and cur_id != prev_id:
                if current:
                    sentences.append(current)
                current, prev_id = [], cur_id
        elif not token and not label:  # Layout B: blank row = sentence break
            if current:
                sentences.append(current)
            current = []
            continue
        if token:
            current.append({"token": token, "tag": normalize_tag(label)})
    if current:
        sentences.append(current)
    logger.info("%s: layout %s, %d sentences", path, "A" if sid else "B", len(sentences))
    return sentences


def split_dev(sentences: list[Sentence], dev_ratio: float, seed: int) -> tuple[list, list]:
    """Hold out part of train as a dev set (used to tune the ensemble threshold)."""
    shuffled = sentences[:]
    random.Random(seed).shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_ratio))
    return shuffled[n_dev:], shuffled[:n_dev]


def summarize(name: str, sentences: list[Sentence]) -> None:
    tags = Counter(t["tag"] for s in sentences for t in s)
    n_tok = sum(tags.values())
    print(f"{name:>5}: {len(sentences):6d} sentences, {n_tok:7d} tokens | "
          + ", ".join(f"{k} {v / n_tok:.0%}" for k, v in tags.most_common(6)))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=0), encoding="utf-8")


def load_json(path: str | Path) -> list[Sentence]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--token-col")
    ap.add_argument("--tag-col")
    ap.add_argument("--sent-col")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kw = dict(token_col=args.token_col, tag_col=args.tag_col, sent_col=args.sent_col)
    train_all = load_csv(args.train, **kw)
    test = load_csv(args.test, **kw)
    if not test:
        raise SystemExit(
            f"No labelled sentences found in {args.test}. The official LinCE test split has its "
            "POS tags hidden, so use the validation file as your test set instead:\n"
            "  python data_prep.py --train pos_hineng_train.csv --test pos_hineng_validation.csv")
    train, dev = split_dev(train_all, args.dev_ratio, args.seed)

    out = Path(args.out)
    for name, data in (("train", train), ("dev", dev), ("test", test)):
        save_json(data, out / f"{name}.json")
        summarize(name, data)
    print(f"Wrote {out}/train.json, dev.json, test.json")


if __name__ == "__main__":
    main()
