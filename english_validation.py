"""
english_validation.py — New Feature: English POS Validation Layer

Provides a lightweight validation layer for the English translated sentence.
Uses NLTK WordNet to find lexical/POS evidence and flags suspicious predictions 
(e.g., 'morning' predicted as VERB when WordNet only has NOUN).
Does NOT overwrite the original Ensemble predictions.
"""
from __future__ import annotations

import logging
try:
    import nltk
    from nltk.corpus import wordnet as wn
    
    # Safely download wordnet if not present
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
except ImportError:
    wn = None
    logging.warning("NLTK is not installed. WordNet validation will be skipped.")

# Mapping WordNet POS to UD POS
WN_TO_UD = {
    'n': 'NOUN',
    'v': 'VERB',
    'a': 'ADJ',
    's': 'ADJ',
    'r': 'ADV'
}

def get_wordnet_evidence(token: str) -> set[str]:
    """Returns a set of possible UD POS tags for a token according to WordNet."""
    if not wn:
        return set()
    
    synsets = wn.synsets(token)
    evidence = set()
    for syn in synsets:
        pos = syn.pos()
        if pos in WN_TO_UD:
            evidence.add(WN_TO_UD[pos])
            
    # Fallback for common function words which WordNet might not cover well
    lower = token.lower()
    if lower in {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}:
        evidence.update({"DET", "PRON"})
    elif lower in {"and", "but", "or", "so", "because", "if", "when", "while", "though"}:
        evidence.update({"CCONJ", "SCONJ"})
    elif lower in {"in", "on", "at", "to", "for", "with", "by", "from", "of", "about"}:
        evidence.update({"ADP", "PART"})
    elif lower in {"is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "shall", "should", "may", "might", "must"}:
        evidence.update({"VERB", "AUX"})
    elif lower in {"i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them"}:
        evidence.update({"PRON"})
        
    return evidence

def validate_english_pos(ensemble_results: list[dict]) -> list[dict]:
    """
    Validates ensemble predictions against WordNet.
    Returns a list of dicts with validation data.
    """
    validation_results = []
    
    for res in ensemble_results:
        token = res["token"]
        ensemble_pos = res["tag"]
        
        evidence = get_wordnet_evidence(token)
        
        wordnet_evidence_str = " | ".join(sorted(evidence)) if evidence else "None found"
        validated_pos = ensemble_pos  # Default to ensemble
        
        if evidence and ensemble_pos not in evidence:
            # ensemble predicted something WordNet completely disagrees with.
            # e.g., morning -> VERB, but evidence is NOUN
            # We must be careful because UD tags are broad (e.g. PROPN vs NOUN).
            if ensemble_pos == "PROPN" and "NOUN" in evidence:
                pass # acceptable
            elif ensemble_pos == "AUX" and "VERB" in evidence:
                pass # acceptable
            elif ensemble_pos == "PART" and "ADV" in evidence:
                pass # acceptable
            else:
                # Strong contradiction. Provide the most common WordNet POS as validated fallback
                # Ideally, we pick the most common synset POS, but here we can just flag it.
                # Since the instruction asks for Validated POS: NOUN, we pick the first/most common
                if wn:
                    syns = wn.synsets(token)
                    if syns:
                        most_common_wn_pos = syns[0].pos()
                        if most_common_wn_pos in WN_TO_UD:
                            validated_pos = WN_TO_UD[most_common_wn_pos]
                            
        validation_results.append({
            "Token": token,
            "Ensemble POS": ensemble_pos,
            "WordNet Evidence": wordnet_evidence_str,
            "Validated POS": validated_pos
        })
        
    return validation_results
