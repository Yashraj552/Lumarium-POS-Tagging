"""metrics.py — token-level accuracy and per-tag / macro / weighted F1 (no sklearn needed)."""
from __future__ import annotations

from collections import Counter


def score(gold: list[str], pred: list[str]) -> dict:
    if len(gold) != len(pred):
        raise ValueError(f"Length mismatch: {len(gold)} gold vs {len(pred)} predicted")
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter(gold)
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    per_tag = {}
    for tag in sorted(set(gold) | set(pred)):
        prec = tp[tag] / (tp[tag] + fp[tag]) if tp[tag] + fp[tag] else 0.0
        rec = tp[tag] / (tp[tag] + fn[tag]) if tp[tag] + fn[tag] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_tag[tag] = {"precision": prec, "recall": rec, "f1": f1, "support": support[tag]}
    gold_tags = [t for t in per_tag if support[t] > 0]  # macro over tags that exist in gold
    n = len(gold)
    return {
        "accuracy": sum(tp.values()) / n if n else 0.0,
        "macro_f1": sum(per_tag[t]["f1"] for t in gold_tags) / len(gold_tags) if gold_tags else 0.0,
        "weighted_f1": sum(per_tag[t]["f1"] * support[t] for t in gold_tags) / n if n else 0.0,
        "n_tokens": n,
        "per_tag": per_tag,
    }


def confusions(gold: list[str], pred: list[str], top: int = 10) -> list[tuple[str, str, int]]:
    """Most frequent (gold -> predicted) mistakes — great material for the report's error analysis."""
    c = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [(g, p, n) for (g, p), n in c.most_common(top)]
