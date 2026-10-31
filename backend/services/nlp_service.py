"""
backend/services/nlp_service.py — Service layer connecting FastAPI to the existing NLP algorithms.
PRESERVES 100% of the existing NLP models, preprocessing, arbitration, morphology,
language identification, NER, translation, and WordNet validation modules.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from collections import Counter

# Ensure root directory is in sys.path so existing modules import seamlessly
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from components.grammar_guide_data import GRAMMAR_GUIDE, GRAMMAR_CATEGORIES
from components.quiz_generator import generate_practice_questions

logger = logging.getLogger(__name__)

# Lumarium POS tag metadata tailored to the warm, scholarly palette
LUMARIUM_TAG_MAP = {
    "NOUN": {
        "name": "Noun (Naming Word)",
        "desc": "Names a person, place, physical object, or concept.",
        "category": "Nouns",
        "color_class": "pos-noun",
        "accent": "#7A9B76",      # Sage
        "accent_soft": "#E8F0E4",
        "icon": "📦",
    },
    "PROPN": {
        "name": "Proper Noun (Special Name)",
        "desc": "A specific name of a person, city, country, or organization.",
        "category": "Nouns",
        "color_class": "pos-propn",
        "accent": "#5E7E5A",      # Deep Sage
        "accent_soft": "#E1EAD9",
        "icon": "🏷️",
    },
    "VERB": {
        "name": "Verb (Action Word)",
        "desc": "Expresses an action, event, movement, or state.",
        "category": "Verbs",
        "color_class": "pos-verb",
        "accent": "#C98282",      # Rose
        "accent_soft": "#F7E9E7",
        "icon": "⚡",
    },
    "AUX": {
        "name": "Auxiliary (Helping Verb)",
        "desc": "Complements the main verb to show tense, mood, or aspect (e.g. hai, aahe, is, had).",
        "category": "Verbs",
        "color_class": "pos-aux",
        "accent": "#B86B6B",      # Deep Rose
        "accent_soft": "#F4DFDC",
        "icon": "🤝",
    },
    "ADJ": {
        "name": "Adjective (Describing Word)",
        "desc": "Describes or modifies a noun, specifying qualities, color, or condition.",
        "category": "Modifiers",
        "color_class": "pos-adj",
        "accent": "#C5A45D",      # Gold
        "accent_soft": "#F5EEDB",
        "icon": "🎨",
    },
    "ADV": {
        "name": "Adverb (How / When Word)",
        "desc": "Qualifies a verb, adjective, or clause by indicating manner, time, or degree.",
        "category": "Modifiers",
        "color_class": "pos-adv",
        "accent": "#B08F47",      # Deep Gold
        "accent_soft": "#EFE6CE",
        "icon": "⏱️",
    },
    "PRON": {
        "name": "Pronoun (Standing-in Word)",
        "desc": "Substitutes for a noun phrase (e.g. I, you, maine, tu, to, usko).",
        "category": "Nouns",
        "color_class": "pos-pron",
        "accent": "#8B83A8",      # Lavender
        "accent_soft": "#EEEBF5",
        "icon": "👤",
    },
    "ADP": {
        "name": "Adposition / Postposition",
        "desc": "Connects words to show location, direction, or role (e.g. in, on, mein, -la, -ne).",
        "category": "Connectors",
        "color_class": "pos-adp",
        "accent": "#667085",      # Slate
        "accent_soft": "#F2F0E9",
        "icon": "📍",
    },
    "DET": {
        "name": "Determiner (Pointing Word)",
        "desc": "Specifies or identifies the reference of a noun (e.g. the, a, ye, to).",
        "category": "Connectors",
        "color_class": "pos-det",
        "accent": "#667085",
        "accent_soft": "#F2F0E9",
        "icon": "👉",
    },
    "PART": {
        "name": "Particle (Helper Word)",
        "desc": "Adds emphasis, focus, or negation without full lexical inflection (e.g. nahi, na, toh, pan).",
        "category": "Connectors",
        "color_class": "pos-part",
        "accent": "#667085",
        "accent_soft": "#F2F0E9",
        "icon": "✨",
    },
    "CCONJ": {
        "name": "Coordinating Conjunction",
        "desc": "Connects syntactic elements of equal grammatical rank (e.g. and, but, aur, aani, pan).",
        "category": "Connectors",
        "color_class": "pos-cconj",
        "accent": "#667085",
        "accent_soft": "#F2F0E9",
        "icon": "🔗",
    },
    "SCONJ": {
        "name": "Subordinating Conjunction",
        "desc": "Introduces a dependent subordinate clause (e.g. because, if, agar, karan).",
        "category": "Connectors",
        "color_class": "pos-sconj",
        "accent": "#667085",
        "accent_soft": "#F2F0E9",
        "icon": "🪢",
    },
    "NUM": {
        "name": "Numeral (Number Word)",
        "desc": "Expresses a cardinal or ordinal number, count, or quantity.",
        "category": "Other",
        "color_class": "pos-num",
        "accent": "#688D64",
        "accent_soft": "#E4EDE2",
        "icon": "🔢",
    },
    "INTJ": {
        "name": "Interjection (Exclamation)",
        "desc": "An emotive expressive utterance (e.g. wow, arey, ouch, yay).",
        "category": "Other",
        "color_class": "pos-intj",
        "accent": "#BD7272",
        "accent_soft": "#F6E4E2",
        "icon": "🎉",
    },
    "PUNCT": {
        "name": "Punctuation Mark",
        "desc": "Syntactic delimiter or symbol (., ?, !, commas).",
        "category": "Other",
        "color_class": "pos-punct",
        "accent": "#9E9AA7",
        "accent_soft": "#F0EEF2",
        "icon": "🔣",
    },
    "SYM": {
        "name": "Symbol",
        "desc": "Mathematical or currency sign, emoji, or symbol.",
        "category": "Other",
        "color_class": "pos-sym",
        "accent": "#9E9AA7",
        "accent_soft": "#F0EEF2",
        "icon": "🔤",
    },
    "X": {
        "name": "Other / Unclassified",
        "desc": "Foreign, unclassified, or unanalyzed token.",
        "category": "Other",
        "color_class": "pos-x",
        "accent": "#9E9AA7",
        "accent_soft": "#F0EEF2",
        "icon": "❓",
    },
}


class NLPService:
    """Singleton service to load and execute existing NLP pipeline modules."""

    def __init__(self):
        self.rule_tagger = load_rule()
        self.hmm_tagger = load_hmm()
        self.transformer = try_load_transformer()
        self.default_cfg = load_arbiter()
        logger.info(
            "NLPService initialized: Rules=%s, HMM=%s, Transformer=%s",
            RULES_PATH.exists(),
            self.hmm_tagger is not None,
            self.transformer is not None,
        )

    def get_health_status(self) -> dict[str, Any]:
        """Provides model health, calibration status, and device info."""
        device_str = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                device_str = f"CUDA ({torch.cuda.get_device_name(0)})"
        except ImportError:
            pass

        return {
            "status": "ok",
            "models": {
                "rules": {
                    "available": RULES_PATH.exists(),
                    "path": str(RULES_PATH),
                    "description": "Deterministic regular expressions and calibrated morphological affix rules",
                },
                "hmm": {
                    "available": self.hmm_tagger is not None,
                    "path": str(HMM_PATH),
                    "description": "Second-order Hidden Markov Model with TnT suffix probability smoothing",
                },
                "transformer": {
                    "available": self.transformer is not None,
                    "device": device_str,
                    "model_name": "HingRoBERTa (l3cube-pune/hing-roberta)",
                    "checkpoint": "models/hingroberta_pos",
                    "description": "Pre-trained Romanized Hinglish code-mixed RoBERTa with 17-tag token classification head",
                },
            },
            "arbiter_defaults": {
                "threshold": float(self.default_cfg.threshold),
                "fallback": self.default_cfg.fallback,
                "rule_overridable": self.default_cfg.rule_overridable,
            },
        }

    def analyze_sentence(
        self,
        text: str,
        threshold: float | None = None,
        fallback: str | None = None,
        rule_overridable: bool | None = None,
        use_transformer: bool = True,
    ) -> dict[str, Any]:
        """
        Executes the full hybrid POS tagger, machine translation, morphology,
        language identification, NER, and WordNet validation.
        """
        clean_text = text.strip()
        if not clean_text:
            return {"error": "Empty input text."}

        # Build dynamic arbiter configuration
        cfg = ArbiterConfig(
            threshold=float(threshold if threshold is not None else self.default_cfg.threshold),
            fallback=str(fallback if fallback is not None else self.default_cfg.fallback),
            rule_overridable=bool(rule_overridable if rule_overridable is not None else self.default_cfg.rule_overridable),
        )

        active_transformer = self.transformer if use_transformer else None

        # 1. Tokenize using text_utils
        tokens = tokenize(clean_text)

        # 2. Execute HybridTagger
        tagger = HybridTagger(self.rule_tagger, active_transformer, self.hmm_tagger, cfg)
        raw_predictions = tagger.predict(tokens)

        # 3. Transliteration & Translation
        translation_data = {
            "language_label": "Code-Mixed",
            "native_script": "",
            "english": "",
            "english_tags": [],
            "available": False,
            "error": None,
        }
        try:
            language_label, native_script, english_text = analyze(clean_text)
            translation_data["language_label"] = language_label
            translation_data["native_script"] = native_script
            translation_data["english"] = english_text
            translation_data["available"] = True
            try:
                translation_data["english_tags"] = english_pos(english_text)
            except (ImportError, OSError):
                translation_data["english_tags"] = tagger.predict(tokenize(english_text))
        except Exception as exc:
            logger.warning("Translation failed: %s", exc)
            translation_data["error"] = str(exc)

        # 4. Morphological Analysis
        morph_original = analyze_morphology(tokens, raw_predictions, is_english=False)
        morph_english = []
        if translation_data["english"].strip():
            eng_tokens = [t["token"] for t in translation_data["english_tags"]]
            eng_hybrid_preds = tagger.predict(eng_tokens)
            morph_english = analyze_morphology(eng_tokens, eng_hybrid_preds, is_english=True)

        # 5. Token-Level Language Identification
        lang_id_data = identify_languages(tokens)

        # 6. Named Entity Recognition
        ner_original = extract_entities(tokens)
        ner_english = []
        if translation_data["english"].strip():
            eng_tokens = [t["token"] for t in translation_data["english_tags"]]
            ner_english = extract_entities(eng_tokens)

        # 7. WordNet Validation Layer
        wordnet_validation = []
        if translation_data["english"].strip():
            eng_tokens = [t["token"] for t in translation_data["english_tags"]]
            eng_hybrid_preds = tagger.predict(eng_tokens)
            wordnet_validation = validate_english_pos(eng_hybrid_preds)

        # 8. Enrich tokens with Lumarium friendly metadata
        enriched_tokens = []
        for r in raw_predictions:
            meta = LUMARIUM_TAG_MAP.get(r["tag"], LUMARIUM_TAG_MAP["X"])
            enriched_tokens.append({
                "word": r["token"],
                "pos": r["tag"],
                "friendly_name": meta["name"],
                "friendly_desc": meta["desc"],
                "category": meta["category"],
                "color_class": meta["color_class"],
                "accent": meta["accent"],
                "accent_soft": meta["accent_soft"],
                "icon": meta["icon"],
                "confidence": r["conf"],
                "source": r["source"],
                "reason": r["reason"],
                "candidates": r["candidates"],
            })

        # 9. Summary & Attribution Metrics
        tag_counts = dict(Counter(r["tag"] for r in raw_predictions))
        source_counts = dict(Counter(r["source"] for r in raw_predictions))

        summary = {
            "total_tokens": len(tokens),
            "tag_counts": tag_counts,
            "source_counts": source_counts,
            "friendly_counts": [
                {
                    "tag": t,
                    "name": LUMARIUM_TAG_MAP.get(t, LUMARIUM_TAG_MAP["X"])["name"],
                    "icon": LUMARIUM_TAG_MAP.get(t, LUMARIUM_TAG_MAP["X"])["icon"],
                    "count": cnt,
                    "accent": LUMARIUM_TAG_MAP.get(t, LUMARIUM_TAG_MAP["X"])["accent"],
                    "accent_soft": LUMARIUM_TAG_MAP.get(t, LUMARIUM_TAG_MAP["X"])["accent_soft"],
                }
                for t, cnt in tag_counts.items()
            ],
        }

        # 10. Technical Decision Trace Table (for the NLP Observatory)
        decision_trace = []
        for r in raw_predictions:
            decision_trace.append({
                "token": r["token"],
                "final_tag": r["tag"],
                "source": r["source"],
                "confidence": r["conf"] if r["conf"] is not None else 0.0,
                "rule_candidate": (
                    f"{r['candidates']['rule']['tag']} ({r['candidates']['rule']['conf']:.2f})"
                    if r["candidates"].get("rule")
                    else "—"
                ),
                "transformer_candidate": (
                    f"{r['candidates']['transformer']['tag']} ({r['candidates']['transformer']['conf']:.2f})"
                    if r["candidates"].get("transformer")
                    else "—"
                ),
                "hmm_candidate": (
                    f"{r['candidates']['hmm']['tag']} ({r['candidates']['hmm']['conf']:.2f})"
                    if r["candidates"].get("hmm")
                    else "—"
                ),
                "reason": r["reason"],
            })

        return {
            "text": clean_text,
            "tokens": enriched_tokens,
            "translation": translation_data,
            "morphology": {
                "original": morph_original,
                "english": morph_english,
            },
            "language_identification": lang_id_data,
            "entities": {
                "original": ner_original,
                "english": ner_english,
            },
            "validation": wordnet_validation,
            "decision_trace": decision_trace,
            "summary": summary,
            "arbiter_settings": {
                "threshold": cfg.threshold,
                "fallback": cfg.fallback,
                "rule_overridable": cfg.rule_overridable,
                "use_transformer": use_transformer,
            },
        }

    def generate_quiz(self, analysis_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Generates dynamic practice questions based on actual NLP output."""
        return generate_practice_questions(analysis_result)

    def get_grammar_guide(self) -> dict[str, Any]:
        """Returns the full compendium of Universal Dependencies grammar cards."""
        return {
            "categories": GRAMMAR_CATEGORIES,
            "guide": GRAMMAR_GUIDE,
        }


# Global singleton instance
nlp_service = NLPService()
