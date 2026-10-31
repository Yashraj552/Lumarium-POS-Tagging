"""
evaluate.py — Week 4: final numbers for the report, on test.json.

    python evaluate.py                  # uses models/ + cached/available transformer
    python evaluate.py --no-transformer

Writes results/report.md (tables ready to paste) and results/metrics.json.
The ablation table is the key result: it shows what each tier contributes and
whether the hybrid actually beats its best single component.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data_prep import load_json
from ensemble import ArbiterConfig, decide
from metrics import confusions, score
from pipeline import MODELS, load_arbiter, load_hmm, load_rule, transformer_predictions


def run_config(cfg, rules, trfs, hmms) -> list[list[dict]]:
    return [[decide(r, t, h, cfg) for r, t, h in zip(rs, ts, hs)]
            for rs, ts, hs in zip(rules, trfs, hmms)]


def flat(nested, key="tag"):
    return [x[key] for s in nested for x in s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--transformer-model")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    test = load_json(args.test)
    tokens = [[x["token"] for x in s] for s in test]
    gold = [g for s in test for g in (x["tag"] for x in s)]

    rule_t, hmm_t, tuned = load_rule(), load_hmm(), load_arbiter()
    if hmm_t is None:
        raise SystemExit("models/hmm.npz missing — run train.py first")
    rule_p = [rule_t.predict_sentence(t) for t in tokens]
    hmm_p = [hmm_t.predict(t) for t in tokens]
    trf_p = None if args.no_transformer else transformer_predictions(
        test, MODELS / "transformer_test.json", args.transformer_model)
    has_trf = trf_p is not None
    if not has_trf:
        trf_p = [[None] * len(s) for s in test]

    rows: list[tuple[str, dict]] = []

    # Rules alone: report coverage + precision (they abstain on most tokens).
    fired = [(r["tag"], g) for r, g in zip((x for s in rule_p for x in s), gold) if r is not None]
    rule_cov = len(fired) / len(gold)
    rule_prec = sum(p == g for p, g in fired) / max(len(fired), 1)

    rows.append(("HMM only", score(gold, flat(hmm_p))))
    if has_trf:
        rows.append(("Transformer only", score(gold, flat(trf_p))))
    rows.append(("Rules → HMM", score(gold, flat(run_config(
        ArbiterConfig(threshold=2.0), rule_p, [[None] * len(s) for s in test], hmm_p)))))
    blueprint_cfg = ArbiterConfig(threshold=0.85, fallback="hmm", rule_overridable=False)
    if has_trf:
        rows.append(("Hybrid — blueprint (0.85, rules final)",
                     score(gold, flat(run_config(blueprint_cfg, rule_p, trf_p, hmm_p)))))
    final = run_config(tuned, rule_p, trf_p, hmm_p)
    final_scores = score(gold, flat(final))
    rows.append((f"Hybrid — tuned on dev ({tuned.threshold:.2f}, {tuned.fallback})", final_scores))

    # Who decided, and how often were they right?
    by_source: dict[str, Counter] = defaultdict(Counter)
    for pred, g in zip((x for s in final for x in s), gold):
        by_source[pred["source"]]["n"] += 1
        by_source[pred["source"]]["correct"] += pred["tag"] == g

    oov = [(h["tag"] == g) for h, g in zip((x for s in hmm_p for x in s), gold) if h["oov"]]

    # ---------------- report ----------------
    lines = ["# Evaluation on test set", "",
             f"{len(test)} sentences, {len(gold)} tokens.", "",
             "## Ablation", "", "| System | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---|---|---|"]
    lines += [f"| {n} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} | {s['weighted_f1']:.4f} |" for n, s in rows]
    lines += ["", f"Rules alone fire on {rule_cov:.1%} of tokens with {rule_prec:.1%} precision.",
              f"HMM accuracy on OOV tokens: {sum(oov) / max(len(oov), 1):.1%} ({len(oov)} tokens).", "",
              "## Which tier made the final call", "", "| Source | Tokens | Share | Accuracy |", "|---|---|---|---|"]
    for src, c in sorted(by_source.items(), key=lambda x: -x[1]["n"]):
        lines.append(f"| {src} | {c['n']} | {c['n'] / len(gold):.1%} | {c['correct'] / c['n']:.1%} |")
    lines += ["", "## Per-tag scores (final hybrid)", "", "| Tag | Precision | Recall | F1 | Support |",
              "|---|---|---|---|---|"]
    for tag, m in sorted(final_scores["per_tag"].items(), key=lambda x: -x[1]["support"]):
        lines.append(f"| {tag} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines += ["", "## Most common errors (gold → predicted)", "", "| Gold | Predicted | Count |", "|---|---|---|"]
    lines += [f"| {g} | {p} | {n} |" for g, p, n in confusions(gold, flat(final))]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(
        {name: {k: v for k, v in s.items()} for name, s in rows}, indent=2), encoding="utf-8")
    print("\n".join(lines[:lines.index("## Which tier made the final call")]))
    print(f"\nFull report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
