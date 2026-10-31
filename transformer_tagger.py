"""
transformer_tagger.py — Tier 2: pre-trained code-mixed BERT token classifier.

Why not `pipeline("token-classification")`?
    The pipeline re-tokenizes raw text and returns sub-word/entity spans, so
    mapping back to OUR word list is fragile (hyphens, emoji and repeated words
    all break character-offset matching). Instead we give the tokenizer the
    word list directly (`is_split_into_words=True`) and use `word_ids()`, which
    says exactly which word every sub-word came from. The tag of a word is the
    prediction on its FIRST sub-word, as the blueprint specifies.

Also handled:
  * sentences longer than the model's 512-token limit (processed in windows);
  * words the tokenizer turns into zero sub-words (rare control chars) -> "X", conf 0;
  * model labels mapped into the shared UD tag set via tagset.normalize_tag.

Data contract:
    predict(tokens: list[str]) -> [{"token", "tag", "source": "transformer", "conf"}, ...]
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from tagset import normalize_tag
from text_utils import normalize_token

logger = logging.getLogger(__name__)

from pathlib import Path
DEFAULT_MODEL = (
    "models/hingroberta_pos" if Path("models/hingroberta_pos").exists()
    else ("models/indicbert_pos" if Path("models/indicbert_pos").exists()
    else "sagorsarker/codeswitch-hineng-pos-lince")
)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def align_first_subword(word_ids: Sequence[int | None], probs: np.ndarray,
                        n_words: int) -> list[tuple[int, float] | None]:
    """
    Pure alignment logic (no torch needed, so it's unit-testable).

    word_ids: for each sub-word position, the index of the word it belongs to
              (None for [CLS]/[SEP]/padding) — exactly what HF's `word_ids()` gives.
    probs:    (n_subwords, n_labels) softmax probabilities.
    Returns one (label_id, prob) per word, or None if the word got no sub-words.
    """
    out: list[tuple[int, float] | None] = [None] * n_words
    for pos, wid in enumerate(word_ids):
        if wid is None or wid >= n_words or out[wid] is not None:
            continue  # special token, or not the first sub-word of this word
        label = int(np.argmax(probs[pos]))
        out[wid] = (label, float(probs[pos, label]))
    return out


class TransformerTagger:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 max_subwords: int = 510) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s", model_name, self.device)

        def _load_hf(loader, *args, **kwargs):
            try:
                return loader.from_pretrained(*args, local_files_only=True, **kwargs)
            except Exception:
                return loader.from_pretrained(*args, **kwargs)

        try:
            self.tokenizer = _load_hf(AutoTokenizer, model_name, use_fast=True)
            self.tokenizer(["probe"], is_split_into_words=True)
        except Exception:  # RoBERTa/XLM-R style tokenizers need this for pre-split input
            self.tokenizer = _load_hf(AutoTokenizer, model_name, use_fast=True,
                                      add_prefix_space=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("A *fast* tokenizer is required for word_ids() alignment.")
        self.model = _load_hf(AutoModelForTokenClassification, model_name).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.id2tag = {i: normalize_tag(lbl) for i, lbl in self.id2label.items()}
        self.max_subwords = max_subwords
        logger.info("Model labels -> UD: %s", {self.id2label[i]: t for i, t in self.id2tag.items()})

    # ------------------------------------------------------------------
    def _windows(self, tokens: list[str]) -> list[tuple[int, int]]:
        """Split a word list into [start, end) windows that fit the sub-word budget."""
        lengths = [max(1, len(self.tokenizer.tokenize(t))) for t in tokens]
        windows, start, used = [], 0, 0
        for i, n in enumerate(lengths):
            if used + n > self.max_subwords and i > start:
                windows.append((start, i))
                start, used = i, 0
            used += n
        windows.append((start, len(tokens)))
        return windows

    def _predict_window(self, words: list[str]) -> list[tuple[int, float] | None]:
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True,
                             max_length=self.max_subwords + 2, return_tensors="pt")
        word_ids = enc.word_ids(0)
        with self.torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        probs = softmax(logits.float().cpu().numpy())
        return align_first_subword(word_ids, probs, len(words))

    def predict(self, tokens: list[str] | str) -> list[dict]:
        if isinstance(tokens, str):
            tokens = [tokens]
        if not tokens:
            return []
        clean_tokens = [
            normalize_token(t) if not (t.startswith("@") or t.startswith("#") or t.startswith("http")) else t
            for t in tokens
        ]
        aligned: list[tuple[int, float] | None] = []
        for start, end in (self._windows(clean_tokens) if len(clean_tokens) > 50 else [(0, len(clean_tokens))]):
            aligned.extend(self._predict_window(clean_tokens[start:end]))
        results = []
        for tok, hit in zip(tokens, aligned):
            if hit is None:
                results.append({"token": tok, "tag": "X", "source": "transformer", "conf": 0.0})
            else:
                label_id, p = hit
                results.append({"token": tok, "tag": self.id2tag[label_id], "source": "transformer",
                                "conf": round(p, 4)})
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tagger = TransformerTagger()
    sentence = "Kal maine dog-ne ka video dekha , he was running sooooo fast yaar 😂".split()
    enc = tagger.tokenizer(sentence, is_split_into_words=True)
    print("Sub-words:", tagger.tokenizer.convert_ids_to_tokens(enc["input_ids"]))
    print("word_ids :", enc.word_ids())
    print()
    for row in tagger.predict(sentence):
        print(row)
