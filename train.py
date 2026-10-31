"""
train.py — build every model in one command.

    python train.py                     # rules + HMM + transformer-based tuning
    python train.py --no-transformer    # laptop without torch: rules + HMM only

Steps
  1. Calibrate the rule tagger on train.json      -> models/rules.json
  2. Train the HMM on train.json                  -> models/hmm.npz
  3. Run all tiers on dev.json, grid-search the
     arbiter threshold + fallback mode            -> models/arbiter.json
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_prep import load_json
from ensemble import ArbiterConfig, tune
from hmm_tagger import HMMTagger
from metrics import score
from pipeline import ARBITER_PATH, HMM_PATH, MODELS, RULES_PATH, transformer_predictions
from rule_tagger import RuleBasedTagger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.json")
    ap.add_argument("--dev", default="data/dev.json")
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--transformer-model")
    ap.add_argument("--alpha-emit", type=float, default=0.01)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train, dev = load_json(args.train), load_json(args.dev)
    gold_dev = [[x["tag"] for x in s] for s in dev]
    tokens_dev = [[x["token"] for x in s] for s in dev]

    print("\n== 1. Calibrating rules ==")
    rules = RuleBasedTagger()
    for name, stats in rules.calibrate(train).items():
        print(f"  {name:22s} {stats}")
    rules.save(RULES_PATH)

    print("\n== 2. Training HMM ==")
    hmm = HMMTagger(alpha_emit=args.alpha_emit).fit(train)
    hmm.save(HMM_PATH)
    hmm_dev = [hmm.predict(t) for t in tokens_dev]
    flat_gold = [g for s in gold_dev for g in s]
    acc = score(flat_gold, [x["tag"] for s in hmm_dev for x in s])["accuracy"]
    print(f"  HMM dev accuracy: {acc:.4f}")

    print("\n== 3. Tuning the arbiter on dev ==")
    rule_dev = [rules.predict_sentence(t) for t in tokens_dev]
    trf_dev = None
    if not args.no_transformer:
        trf_dev = transformer_predictions(dev, MODELS / "transformer_dev.json", args.transformer_model)
    if trf_dev is None:
        print("  No transformer available — tuning rules + HMM only.")
        trf_dev = [[None] * len(s) for s in dev]
    else:
        acc = score(flat_gold, [x["tag"] for s in trf_dev for x in s])["accuracy"]
        print(f"  Transformer dev accuracy: {acc:.4f}")

    best, best_acc, results = tune(zip(rule_dev, trf_dev, hmm_dev, gold_dev))
    blueprint = next(a for c, a in results
                     if abs(c.threshold - 0.85) < 1e-9 and c.fallback == "hmm" and c.rule_overridable is False)
    print(f"  Blueprint-style config (th=0.85, HMM fallback, rules final): dev acc {blueprint:.4f}")
    print(f"  Best config: {best}  dev acc {best_acc:.4f}")
    best.save(ARBITER_PATH)
    print(f"\nSaved {RULES_PATH}, {HMM_PATH}, {ARBITER_PATH}")


if __name__ == "__main__":
    main()
