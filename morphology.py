"""
morphology.py — New Feature 1: Morphological Analysis

Extracts linguistic information like Root, Prefix, Suffix, Number, and Case where possible.
Reuses existing lexicons and rules from rule_tagger.py.
"""
from __future__ import annotations

import re
from rule_tagger import REGIONAL_CASE_MARKERS, LIGHT_VERBS, MARATHI_FUNCTION_WORDS
from text_utils import normalize_token

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    lemmatizer = WordNetLemmatizer()
except ImportError:
    wn = None
    lemmatizer = None

def analyze_morphology(tokens: list[str], pos_results: list[dict] = None, is_english: bool = False) -> list[dict]:
    """
    Returns a list of dictionaries with morphological features for each token.
    """
    results = []
    
    # Optional POS context
    pos_map = {res["token"]: res["tag"] for res in pos_results} if pos_results else {}

    for token in tokens:
        norm = normalize_token(token)
        pos = pos_map.get(token, "Unknown")
        
        info = {
            "Token": token,
            "Root": token,
            "Prefix": "Not detected",
            "Suffix": "Not detected",
            "Number": "Unknown",
            "Case": "Unknown",
            "Other": "Unknown"
        }
        
        # Heuristics based on existing Rule Tagger logic
        
        # 1. Check for hyphenated case markers (e.g., dog-ne)
        if "-" in norm:
            parts = norm.split("-")
            if len(parts) == 2 and parts[1] in REGIONAL_CASE_MARKERS:
                info["Root"] = token.rsplit("-", 1)[0]
                info["Suffix"] = "-" + parts[1]
                info["Case"] = "Oblique/Marker"
        
        # 2. Check for fused case markers
        else:
            for marker in sorted(REGIONAL_CASE_MARKERS, key=len, reverse=True):
                if norm.endswith(marker) and len(norm) > len(marker):
                    info["Root"] = token[:-len(marker)]
                    info["Suffix"] = token[-len(marker):]
                    info["Case"] = "Oblique/Marker"
                    break
        
        # 3. Hindi and English plural forms (heuristics)
        if norm.endswith(("on", "aan", "iyan", "ein")) and pos in ("NOUN", "PROPN") and len(norm) > 3:
            info["Number"] = "Plural"
            if info["Suffix"] == "Not detected":
                info["Suffix"] = token[-2:]
                info["Root"] = token[:-2]
            
        elif norm.endswith(("s", "es", "ies")) and pos in ("NOUN", "PROPN") and info["Root"] == token:
             info["Number"] = "Plural"
             if norm.endswith("ies") and len(norm) > 3:
                info["Suffix"] = token[-3:]
                info["Root"] = token[:-3]
             elif norm.endswith("es") and len(norm) > 2:
                info["Suffix"] = token[-2:]
                info["Root"] = token[:-2]
             elif norm.endswith("s") and len(norm) > 1:
                info["Suffix"] = token[-1:]
                info["Root"] = token[:-1]
        elif info["Number"] == "Unknown" and pos in ("NOUN", "PROPN"):
             info["Number"] = "Singular"
             
        # 4. Light verbs
        if norm in LIGHT_VERBS:
            info["Other"] = "Light Verb"
            
        # 5. Verb Tense Heuristics (Hindi)
        if pos == "VERB" or pos == "AUX":
            tense = None
            if norm.endswith(("ga", "gi", "ge")):
                tense = "Future Tense"
            elif norm.endswith(("ta", "ti", "te")):
                tense = "Present Tense"
            elif norm.endswith(("ya", "yi", "ye", "aa", "ee")) or norm in ("tha", "thi", "the", "gaya", "gaye", "gayi"):
                tense = "Past Tense"
            
            if tense:
                if info["Other"] == "Unknown":
                    info["Other"] = tense
                else:
                    info["Other"] = info["Other"] + " + " + tense

        # 6. WordNet logic for English tokens
        if is_english and wn and lemmatizer:
            # Get lemma based on simple heuristic or WordNet POS mapping
            pos_tag = 'n'
            if pos.startswith('V'): pos_tag = 'v'
            elif pos.startswith('ADJ'): pos_tag = 'a'
            elif pos.startswith('ADV'): pos_tag = 'r'
            
            lemma = lemmatizer.lemmatize(token.lower(), pos=pos_tag)
            if lemma and lemma != token.lower():
                info["Root"] = lemma
            
            synsets = wn.synsets(token)
            if synsets:
                lexical_cats = list(set([s.pos() for s in synsets]))
                cats_str = "WN:" + ",".join(lexical_cats)
                if info["Other"] == "Unknown":
                    info["Other"] = cats_str
                else:
                    info["Other"] += " | " + cats_str

        results.append(info)
        
    return results
