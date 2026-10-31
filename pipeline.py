"""
pipeline.py — loading helpers shared by train.py, evaluate.py and app.py.

Transformer predictions can be cached to JSON. That lets Member 3 run the
heavy model once on Colab's GPU, download `models/transformer_dev.json` and
`models/transformer_test.json`, and everyone else evaluates on a laptop
without installing torch.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ensemble import ArbiterConfig
from hmm_tagger import HMMTagger
from rule_tagger import RuleBasedTagger

logger = logging.getLogger(__name__)

MODELS = Path("models")
RULES_PATH = MODELS / "rules.json"
HMM_PATH = MODELS / "hmm.npz"
ARBITER_PATH = MODELS / "arbiter.json"


def load_rule(path: Path = RULES_PATH) -> RuleBasedTagger:
    if path.exists():
        return RuleBasedTagger.load(path)
    logger.warning("%s not found — using uncalibrated default rules", path)
    return RuleBasedTagger()


def load_hmm(path: Path = HMM_PATH) -> HMMTagger | None:
    if path.exists():
        return HMMTagger.load(path)
    logger.warning("%s not found — run train.py first. HMM tier disabled.", path)
    return None


def load_arbiter(path: Path = ARBITER_PATH) -> ArbiterConfig:
    return ArbiterConfig.load(path) if path.exists() else ArbiterConfig()


def try_load_transformer(model_name: str | None = None):
    """Return a TransformerTagger, or None (with the reason logged) if torch/model is unavailable."""
    try:
        from transformer_tagger import DEFAULT_MODEL, TransformerTagger
        return TransformerTagger(model_name or DEFAULT_MODEL)
    except Exception as exc:  # ImportError, no internet, OOM...
        logger.warning("Transformer tier disabled: %s", exc)
        return None


def transformer_predictions(sentences: list[list[dict]], cache: Path, model_name: str | None = None,
                            tagger=None) -> list[list[dict]] | None:
    """Predictions for every sentence, from cache if valid, else computed (and cached)."""
    if cache.exists():
        preds = json.loads(cache.read_text(encoding="utf-8"))
        if len(preds) == len(sentences) and all(len(p) == len(s) for p, s in zip(preds, sentences)):
            logger.info("Using cached transformer predictions from %s", cache)
            return preds
        logger.warning("Cache %s doesn't match the data — recomputing", cache)
    tagger = tagger or try_load_transformer(model_name)
    if tagger is None:
        return None
    preds = []
    for i, sent in enumerate(sentences):
        preds.append(tagger.predict([x["token"] for x in sent]))
        if (i + 1) % 200 == 0:
            logger.info("transformer: %d / %d sentences", i + 1, len(sentences))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    return preds
