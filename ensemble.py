"""
ensemble.py — Member 4: the Arbiter that decides which tier to trust per token.

Kept separate from the Streamlit app so it can be unit-tested and reused by
evaluate.py (the numbers in your report come from exactly the code the UI runs).

Decision logic per token (waterfall, as in the blueprint, with two fixes):

  1. Rule fired?                -> accept it ...
       ... unless the transformer disagrees with HIGHER confidence than the
       rule's calibrated precision (rules are no longer blindly 1.0).
  2. Transformer conf >= threshold -> accept transformer.
  3. Otherwise fall back. Two fallback modes, picked on dev.json by tune():
       "hmm"      — blueprint behaviour: HMM always wins the fallback.
       "max_conf" — whichever of transformer / HMM is more confident.
     If transformer and HMM agree, that tag is taken either way.

The threshold (0.85 in the blueprint) is tuned on dev.json, never on test.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class ArbiterConfig:
    threshold: float = 0.85
    fallback: str = "hmm"               # "hmm" | "max_conf"
    rule_overridable: bool = True

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ArbiterConfig":
        return cls(**json.loads(Path(path).read_text()))


def decide(rule: dict | None, trf: dict | None, hmm: dict | None, cfg: ArbiterConfig) -> dict:
    """Pick one prediction for one token. Every input follows the data contract (or is None)."""
    token = next(x["token"] for x in (rule, trf, hmm) if x is not None)

    def pick(src: dict, reason: str, source: str | None = None, conf: float | None = None) -> dict:
        return {"token": token, "tag": src["tag"], "source": source or src["source"],
                "conf": round(src["conf"] if conf is None else conf, 4), "reason": reason}

    if rule is not None:
        if (cfg.rule_overridable and trf is not None and trf["tag"] != rule["tag"]
                and trf["conf"] > rule["conf"]):
            return pick(trf, f"transformer ({trf['conf']:.2f}) outranked rule ({rule['conf']:.2f})")
        return pick(rule, f"rule '{rule.get('rule', '?')}' fired")

    if trf is not None and trf["conf"] >= cfg.threshold:
        return pick(trf, f"transformer confident (≥ {cfg.threshold:.2f})")

    if hmm is None:  # HMM missing: best we can do
        return pick(trf, "transformer (no HMM available)")
    if trf is None:
        return pick(hmm, "HMM (no transformer available)")
    if trf["tag"] == hmm["tag"]:
        return pick(trf, "transformer and HMM agree", source="transformer+hmm",
                    conf=max(trf["conf"], hmm["conf"]))
    # When HMM is OOV, the HMM has never observed this word: prefer the pretrained contextual transformer
    if hmm.get("oov", False):
        return pick(trf, "HMM is OOV; trusting transformer")
    if cfg.fallback == "max_conf" and trf["conf"] >= hmm["conf"]:
        return pick(trf, "low-confidence transformer still beat HMM")
    return pick(hmm, "fallback to HMM")


class HybridTagger:
    """Wraps the three tiers. Any tier may be None (e.g. no GPU / no torch installed)."""

    def __init__(self, rule_tagger=None, transformer_tagger=None, hmm_tagger=None,
                 config: ArbiterConfig | None = None) -> None:
        if not any((rule_tagger, transformer_tagger, hmm_tagger)):
            raise ValueError("At least one tagger is required")
        self.rule = rule_tagger
        self.transformer = transformer_tagger
        self.hmm = hmm_tagger
        self.config = config or ArbiterConfig()

    def candidates(self, tokens: list[str], transformer_out: list[dict] | None = None
                   ) -> tuple[list, list, list]:
        """Run every available tier once over the WHOLE sentence (HMM/BERT need context)."""
        n = len(tokens)
        rules = self.rule.predict_sentence(tokens) if self.rule else [None] * n
        if transformer_out is None:
            transformer_out = self.transformer.predict(tokens) if self.transformer else [None] * n
        hmms = self.hmm.predict(tokens) if self.hmm else [None] * n
        return rules, transformer_out, hmms

    def predict(self, tokens: list[str], transformer_out: list[dict] | None = None) -> list[dict]:
        if not tokens:
            return []
        rules, trfs, hmms = self.candidates(tokens, transformer_out)
        out = []
        for r, t, h in zip(rules, trfs, hmms):
            final = decide(r, t, h, self.config)
            final["candidates"] = {"rule": r, "transformer": t, "hmm": h}
            out.append(final)
        return out


def tune(cached: Iterable[tuple[list, list, list, list[str]]],
         thresholds: Iterable[float] = tuple(sorted({x / 100 for x in range(50, 100, 2)} | {0.85})),
         ) -> tuple[ArbiterConfig, float, list[tuple[ArbiterConfig, float]]]:
    """
    Grid-search the arbiter on dev data.
    cached: per sentence, (rule_preds, transformer_preds, hmm_preds, gold_tags) —
            computed ONCE, so the grid search itself is instant.
    Returns (best_config, best_accuracy, all_results).
    """
    cached = list(cached)
    results = []
    for fallback in ("hmm", "max_conf"):
        for overridable in (True, False):
            for th in thresholds:
                cfg = ArbiterConfig(threshold=th, fallback=fallback, rule_overridable=overridable)
                correct = total = 0
                for rules, trfs, hmms, gold in cached:
                    for r, t, h, g in zip(rules, trfs, hmms, gold):
                        correct += decide(r, t, h, cfg)["tag"] == g
                        total += 1
                results.append((cfg, correct / max(total, 1)))
    best_cfg, best_acc = max(results, key=lambda x: x[1])
    return best_cfg, best_acc, results
