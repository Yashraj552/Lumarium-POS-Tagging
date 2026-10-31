"""
nlp_bridge.py — Clean integration layer between Grammar Buddy's UI and the NLP engine.
Preserves 100% of the underlying models, arbitration, morphology, LID, NER,
and translation algorithms while providing structured data objects for the frontend.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ensemble import ArbiterConfig, HybridTagger
from pipeline import (
    ARBITER_PATH,
    HMM_PATH,
    RULES_PATH,
    load_arbiter,
    load_hmm,
    load_rule,
    try_load_transformer,
)
from text_utils import tokenize
from translate import analyze, english_pos
from morphology import analyze_morphology
from language_identifier import identify_languages
from ner import extract_entities
from english_validation import validate_english_pos

logger = logging.getLogger(__name__)


@dataclass
class ModelInventory:
    rule_tagger: Any
    hmm_tagger: Any
    transformer: Any
    saved_cfg: ArbiterConfig
    rules_ok: bool
    hmm_ok: bool
    transformer_ok: bool


def get_model_inventory(use_transformer: bool = True) -> ModelInventory:
    """Load and return all available backend model tiers and configurations."""
    rule_tagger = load_rule()
    hmm_tagger = load_hmm()
    transformer = try_load_transformer() if use_transformer else None
    saved_cfg = load_arbiter()

    return ModelInventory(
        rule_tagger=rule_tagger,
        hmm_tagger=hmm_tagger,
        transformer=transformer,
        saved_cfg=saved_cfg,
        rules_ok=RULES_PATH.exists(),
        hmm_ok=hmm_tagger is not None,
        transformer_ok=transformer is not None,
    )


def process_sentence(
    text: str,
    arbiter_cfg: ArbiterConfig,
    inventory: ModelInventory,
) -> dict[str, Any]:
    """
    Executes the full NLP pipeline on the user's sentence without modifying any model logic.
    Returns a unified structured dictionary.
    """
    clean_text = text.strip()
    if not clean_text:
        return {}

    tokens = tokenize(clean_text)

    # 1. Run the core Hybrid POS Tagger
    tagger = HybridTagger(
        inventory.rule_tagger,
        inventory.transformer,
        inventory.hmm_tagger,
        arbiter_cfg,
    )
    pos_results = tagger.predict(tokens)

    # 2. Transliteration and Translation
    translation_info = {
        "language_label": "Code-Mixed",
        "native_script": "",
        "english": "",
        "english_tags": [],
        "available": False,
        "error": None,
    }
    try:
        language, native, english = analyze(clean_text)
        translation_info["language_label"] = language
        translation_info["native_script"] = native
        translation_info["english"] = english
        translation_info["available"] = True
        try:
            translation_info["english_tags"] = english_pos(english)
        except (ImportError, OSError):
            translation_info["english_tags"] = tagger.predict(tokenize(english))
    except Exception as exc:
        logger.warning("Translation pipeline failed: %s", exc)
        translation_info["error"] = str(exc)

    # 3. Morphological Analysis
    morph_original = analyze_morphology(tokens, pos_results, is_english=False)
    morph_english = []
    if translation_info["english"].strip():
        eng_tokens = [t["token"] for t in translation_info["english_tags"]]
        eng_hybrid_preds = tagger.predict(eng_tokens)
        morph_english = analyze_morphology(eng_tokens, eng_hybrid_preds, is_english=True)

    # 4. Token-Level Language Identification (LID)
    lang_id_data = identify_languages(tokens)

    # 5. Named Entity Recognition (NER)
    ner_original = extract_entities(tokens)
    ner_english = []
    if translation_info["english"].strip():
        eng_tokens = [t["token"] for t in translation_info["english_tags"]]
        ner_english = extract_entities(eng_tokens)

    # 6. English POS Validation Layer (WordNet)
    validation_results = []
    if translation_info["english"].strip():
        eng_tokens = [t["token"] for t in translation_info["english_tags"]]
        eng_hybrid_preds = tagger.predict(eng_tokens)
        validation_results = validate_english_pos(eng_hybrid_preds)

    # 7. Summary metrics
    tag_counts = Counter(r["tag"] for r in pos_results)

    return {
        "raw_text": clean_text,
        "tokens": tokens,
        "pos_results": pos_results,
        "translation": translation_info,
        "morphology": {
            "original": morph_original,
            "english": morph_english,
        },
        "language_id": lang_id_data,
        "entities": {
            "original": ner_original,
            "english": ner_english,
        },
        "validation": validation_results,
        "tag_counts": dict(tag_counts),
        "total_tokens": len(tokens),
    }
