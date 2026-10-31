"""
hmm_tagger.py — Tier 3: first-order Hidden Markov Model + Viterbi, NumPy only.

What it does
    fit(corpus)          -> learns transition matrix A (tag->tag, with START/END)
                            and emission matrix B (tag->word), with add-k smoothing
    predict(tokens)      -> Viterbi decoding of the whole sentence

Improvements over the original spec (all explained in the README):
  * Everything runs in log space, so long sentences don't underflow to 0.
  * OOV words don't just get a flat smoothed probability. Like the TnT tagger,
    we estimate P(tag | suffix) from rare training words — a fused form like
    "pakadla" still gets verb-ish scores from its "-dla"/"-la" ending.
    Capitalized and lowercase words use separate suffix tables.
  * URLs / numbers / mentions / emoji share one pseudo-word each (<URL>, ...).
  * Confidence is the true posterior P(tag_i | whole sentence) from the
    forward-backward algorithm, not a normalized emission.

Data contract:
    predict(tokens: list[str]) -> [{"token", "tag", "source": "hmm", "conf"}, ...]
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tagset import normalize_tag
from text_utils import normalize_token, token_shape

logger = logging.getLogger(__name__)

NEG_INF = -1e30


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class HMMTagger:
    def __init__(self, alpha_trans: float = 0.01, alpha_emit: float = 0.01,
                 max_suffix: int = 4, rare_threshold: int = 10, suffix_theta: float = 0.3) -> None:
        """
        alpha_trans / alpha_emit: add-k smoothing constants (1.0 = Laplace).
            Plain Laplace on emissions over a 20k-word vocabulary gives far too
            much mass to unseen pairs; 0.1 is a better default (tune on dev).
        max_suffix / rare_threshold: suffix model for OOV words, learned from
            words seen <= rare_threshold times (they behave most like unseen words).
        """
        self.alpha_trans = alpha_trans
        self.alpha_emit = alpha_emit
        self.max_suffix = max_suffix
        self.rare_threshold = rare_threshold
        self.suffix_theta = suffix_theta
        self.tags: list[str] = []
        self.word_index: dict[str, int] = {}
        self.log_start: np.ndarray | None = None   # (T,)
        self.log_trans: np.ndarray | None = None   # (T, T)  A[i, j] = log P(t_j | t_i)
        self.log_end: np.ndarray | None = None     # (T,)
        self.log_emit: np.ndarray | None = None    # (T, V)  B[t, w] = log P(w | t)
        self.log_emit_unk: np.ndarray | None = None  # (T,) smoothed mass for unseen words
        self.log_prior: np.ndarray | None = None   # (T,) log P(t)
        self.suffix_logp: dict[str, np.ndarray] = {}  # "c|ing" -> log P(t | suffix, cap)

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _key(token: str) -> str:
        return token_shape(token) or normalize_token(token)

    @staticmethod
    def _cap(token: str) -> str:
        return "C" if token[:1].isupper() else "l"

    # ------------------------------------------------------------------ training
    def fit(self, corpus: list[list[dict]]) -> "HMMTagger":
        tag_counts: Counter = Counter()
        word_counts: Counter = Counter()
        start: Counter = Counter()
        end: Counter = Counter()
        trans: Counter = Counter()
        emit: Counter = Counter()
        raw_forms: dict[str, Counter] = defaultdict(Counter)

        for sent in corpus:
            if not sent:
                continue
            tags = [normalize_tag(x["tag"]) for x in sent]
            start[tags[0]] += 1
            end[tags[-1]] += 1
            for a, b in zip(tags, tags[1:]):
                trans[a, b] += 1
            for item, tag in zip(sent, tags):
                key = self._key(item["token"])
                tag_counts[tag] += 1
                word_counts[key] += 1
                emit[tag, key] += 1
                raw_forms[key][item["token"]] += 1

        self.tags = sorted(tag_counts)
        t_idx = {t: i for i, t in enumerate(self.tags)}
        vocab = sorted(word_counts)
        self.word_index = {w: i for i, w in enumerate(vocab)}
        T, V = len(self.tags), len(vocab)
        if T == 0:
            raise ValueError("Empty training corpus")

        # Transitions (+ START row / END column), add-k smoothed.
        a = self.alpha_trans
        start_vec = np.array([start[t] for t in self.tags], float)
        self.log_start = np.log((start_vec + a) / (start_vec.sum() + a * T))
        trans_mat = np.zeros((T, T))
        for (x, y), c in trans.items():
            trans_mat[t_idx[x], t_idx[y]] = c
        end_vec = np.array([end[t] for t in self.tags], float)
        denom = trans_mat.sum(1) + end_vec + a * (T + 1)
        self.log_trans = np.log((trans_mat + a) / denom[:, None])
        self.log_end = np.log((end_vec + a) / denom)

        # Emissions, add-k smoothed with one extra slot reserved for "unseen word".
        k = self.alpha_emit
        emit_mat = np.zeros((T, V), dtype=np.float64)
        for (t, w), c in emit.items():
            emit_mat[t_idx[t], self.word_index[w]] = c
        tag_totals = emit_mat.sum(1)
        denom = tag_totals + k * (V + 1)
        self.log_emit = np.log((emit_mat + k) / denom[:, None]).astype(np.float32)
        self.log_emit_unk = np.log(k / denom)
        self.log_prior = np.log(tag_totals / tag_totals.sum())

        self._fit_suffixes(emit, word_counts, raw_forms, t_idx)
        logger.info("HMM trained: %d tags, %d word types, %d suffix entries",
                    T, V, len(self.suffix_logp))
        return self

    def _fit_suffixes(self, emit: Counter, word_counts: Counter,
                      raw_forms: dict[str, Counter], t_idx: dict[str, int]) -> None:
        """TnT-style P(tag | suffix) from rare words, one table per capitalization class."""
        T = len(self.tags)
        counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(T))
        for (tag, key), c in emit.items():
            if word_counts[key] > self.rare_threshold or key.startswith("<"):
                continue
            cap = self._cap(raw_forms[key].most_common(1)[0][0])
            for n in range(0, min(self.max_suffix, len(key)) + 1):
                counts[f"{cap}|{key[len(key) - n:] if n else ''}"][t_idx[tag]] += c
        self.suffix_logp = {}
        for sfx, vec in counts.items():
            self.suffix_logp[sfx] = vec  # raw counts for now; interpolated below

        # Successive interpolation from shortest to longest suffix (Brants 2000).
        theta = self.suffix_theta
        fallback = np.full(T, 1.0 / T)
        smoothed: dict[str, np.ndarray] = {}
        for sfx in sorted(counts, key=lambda s: len(s)):
            cap, ending = sfx.split("|", 1)
            parent = smoothed.get(f"{cap}|{ending[1:]}") if ending else fallback
            if parent is None:
                parent = fallback
            vec = counts[sfx]
            p_hat = vec / vec.sum() if vec.sum() else parent
            smoothed[sfx] = (p_hat + theta * parent) / (1 + theta)
        self.suffix_logp = {s: np.log(np.maximum(v, 1e-12)) for s, v in smoothed.items()}

    # ------------------------------------------------------------------ emissions
    def _emission_column(self, token: str) -> tuple[np.ndarray, bool]:
        """log P(token | t) for every tag (up to a per-position constant). Returns (vec, known)."""
        key = self._key(token)
        idx = self.word_index.get(key)
        if idx is not None:
            return self.log_emit[:, idx].astype(np.float64), True
        if key.startswith("<"):  # unseen shape class: fall back to generic unk mass
            return self.log_emit_unk.copy(), False
        cap = self._cap(token)
        vec = None
        for n in range(min(self.max_suffix, len(key)), -1, -1):
            sfx = f"{cap}|{key[len(key) - n:] if n else ''}"
            if sfx in self.suffix_logp:
                # Bayes: P(w|t) ∝ P(t|suffix) / P(t)   (P(w) is constant across tags)
                vec = (self.suffix_logp[sfx] - self.log_prior).copy()
                break
        if vec is None:
            vec = self.log_emit_unk.copy()

        # Add linguistic feature boosts for Hinglish OOV tokens
        t_idx = getattr(self, "_t_idx", None)
        if t_idx is None:
            self._t_idx = {t: i for i, t in enumerate(self.tags)}
            t_idx = self._t_idx

        low = token.lower()
        if token.istitle() and token.isalpha() and len(token) > 2 and "PROPN" in t_idx:
            vec[t_idx["PROPN"]] += 1.8
        elif low.endswith(("kar", "karke", "unga", "enge", "ungi", "ega", "egi", "te", "ta", "ti")) and len(low) > 3 and "VERB" in t_idx:
            vec[t_idx["VERB"]] += 1.5
        elif low.endswith(("on", "aan", "iyan", "ein")) and len(low) > 3 and "NOUN" in t_idx:
            vec[t_idx["NOUN"]] += 1.5
        elif low.endswith(("ly", "fully")) and len(low) > 4 and "ADV" in t_idx:
            vec[t_idx["ADV"]] += 2.0

        return vec, False

    # ------------------------------------------------------------------ decoding
    def _lattice(self, tokens: list[str]) -> tuple[np.ndarray, list[bool]]:
        cols = [self._emission_column(t) for t in tokens]
        return np.stack([c[0] for c in cols]), [c[1] for c in cols]

    def viterbi(self, emissions: np.ndarray) -> list[int]:
        n, T = emissions.shape
        score = self.log_start + emissions[0]
        back = np.zeros((n, T), dtype=np.int32)
        for i in range(1, n):
            cand = score[:, None] + self.log_trans          # (prev, cur)
            back[i] = np.argmax(cand, axis=0)
            score = cand[back[i], np.arange(T)] + emissions[i]
        score = score + self.log_end
        path = [int(np.argmax(score))]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i, path[-1]]))
        return path[::-1]

    def posteriors(self, emissions: np.ndarray) -> np.ndarray:
        """Forward-backward: returns (n, T) matrix of P(tag_i = t | sentence)."""
        n, T = emissions.shape
        fwd = np.empty((n, T))
        bwd = np.empty((n, T))
        fwd[0] = self.log_start + emissions[0]
        for i in range(1, n):
            fwd[i] = _logsumexp(fwd[i - 1][:, None] + self.log_trans, axis=0) + emissions[i]
        bwd[-1] = self.log_end
        for i in range(n - 2, -1, -1):
            bwd[i] = _logsumexp(self.log_trans + (emissions[i + 1] + bwd[i + 1])[None, :], axis=1)
        log_post = fwd + bwd
        log_post -= _logsumexp(log_post, axis=1)[:, None]
        return np.exp(log_post)

    def predict(self, tokens: list[str] | str) -> list[dict]:
        if self.log_trans is None:
            raise RuntimeError("HMMTagger is not trained — call fit() or load() first")
        if isinstance(tokens, str):
            tokens = [tokens]
        if not tokens:
            return []
        emissions, known = self._lattice(tokens)
        path = self.viterbi(emissions)
        post = self.posteriors(emissions)
        return [{"token": tok, "tag": self.tags[t], "source": "hmm",
                 "conf": round(float(post[i, t]), 4), "oov": not known[i]}
                for i, (tok, t) in enumerate(zip(tokens, path))]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sfx_keys = sorted(self.suffix_logp)
        np.savez_compressed(
            path,
            tags=np.array(self.tags), vocab=np.array(sorted(self.word_index, key=self.word_index.get)),
            log_start=self.log_start, log_trans=self.log_trans, log_end=self.log_end,
            log_emit=self.log_emit, log_emit_unk=self.log_emit_unk, log_prior=self.log_prior,
            sfx_keys=np.array(sfx_keys, dtype=str),
            sfx_vals=np.stack([self.suffix_logp[k] for k in sfx_keys]) if sfx_keys else np.zeros((0, 0)),
            params=np.array([self.alpha_trans, self.alpha_emit, self.max_suffix,
                             self.rare_threshold, self.suffix_theta]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HMMTagger":
        d = np.load(path, allow_pickle=False)
        p = d["params"]
        m = cls(alpha_trans=float(p[0]), alpha_emit=float(p[1]), max_suffix=int(p[2]),
                rare_threshold=int(p[3]), suffix_theta=float(p[4]))
        m.tags = [str(t) for t in d["tags"]]
        m.word_index = {str(w): i for i, w in enumerate(d["vocab"])}
        for name in ("log_start", "log_trans", "log_end", "log_emit", "log_emit_unk", "log_prior"):
            setattr(m, name, d[name])
        m.suffix_logp = {str(k): v for k, v in zip(d["sfx_keys"], d["sfx_vals"])}
        return m


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) == 4 and sys.argv[1] == "train":
        from data_prep import load_json

        model = HMMTagger().fit(load_json(sys.argv[2]))
        model.save(sys.argv[3])
        print(f"Saved HMM to {sys.argv[3]}")
        sys.exit()

    # Toy demo: proves the lattice decodes and OOV suffixes work.
    def S(text: str) -> list[dict]:
        return [{"token": w.rsplit("/", 1)[0], "tag": w.rsplit("/", 1)[1]} for w in text.split()]

    toy = [
        S("maine/PRON dog/NOUN ko/ADP dekha/VERB ./PUNCT"),
        S("usne/PRON cat/NOUN ko/ADP pakdla/VERB ./PUNCT"),
        S("I/PRON saw/VERB the/DET dog/NOUN yesterday/NOUN ./PUNCT"),
        S("tune/PRON khana/NOUN khaya/VERB kya/PART ?/PUNCT"),
        S("he/PRON quickly/ADV jump/VERB kela/AUX ./PUNCT"),
        S("mala/PRON ghar/NOUN ja/VERB ayla/VERB havay/AUX ./PUNCT"),
        S("the/DET movie/NOUN was/AUX awesome/ADJ yaar/INTJ !/PUNCT"),
    ]
    model = HMMTagger().fit(toy)
    for sent in (["maine", "cat", "ko", "pakadla", "!"], ["the", "phone", "was", "awesome"]):
        for r in model.predict(sent):
            print(r)
        print()
