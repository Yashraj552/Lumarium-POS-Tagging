"""
language_identifier.py — Token-Level Language Identification & Code-Mixing Index (CMI).

Identifies the language of each token in a code-mixed Hindi-English (Hinglish) text
and computes Gambäck & Das (2014, 2016) Code-Mixing Index (CMI) + Switch Points.
"""
from __future__ import annotations

import re
from collections import Counter
from translate import is_english
from text_utils import token_shape

WORD_RE = re.compile(r"^[^\W\d_]+$")


def calculate_cmi(token_lang_pairs: list[dict]) -> dict:
    """
    Computes Gambäck & Das (2014, 2016) Code-Mixing Index (CMI) and Switch Points.
    
    Formula:
      CMI = 100 * (1 - max(w_i) / (n - u))  if (n - u) > 0 else 0
      CMI_all = 0.5 * (CMI + 100 * (SwitchPoints / (n - u - 1)))
      
    Where:
      - (n - u): language-dependent tokens (excluding punctuation, numbers, symbols)
      - max(w_i): word count of the dominant (matrix) language
      - SwitchPoints: transitions between adjacent language tokens
    """
    # Filter for language-specific tokens (exclude 'Other', punct, numbers)
    valid_langs = [
        p["Language"] for p in token_lang_pairs 
        if p.get("Language") in ("Hindi", "English")
    ]
    
    n_lang = len(valid_langs)
    if n_lang == 0:
        return {
            "cmi": 0.0,
            "switch_points": 0,
            "cmi_all": 0.0,
            "dominant_lang": "None",
            "lang_tokens_count": 0,
        }

    lang_counts = Counter(valid_langs)
    dominant_lang, max_count = lang_counts.most_common(1)[0]

    # 1. Base CMI (Gambäck & Das 2014):
    # Monolingual = 0.0%, 50/50 mix = 50.0%
    cmi = 100.0 * (1.0 - (max_count / n_lang))

    # 2. Count Switch Points (transitions between adjacent language tokens)
    switch_points = sum(
        1 for i in range(n_lang - 1) if valid_langs[i] != valid_langs[i + 1]
    )

    # 3. Extended CMI with Switch Points (Gambäck & Das 2016)
    sp_fraction = (switch_points / (n_lang - 1)) if n_lang > 1 else 0.0
    cmi_all = 0.5 * (cmi + (100.0 * sp_fraction))

    return {
        "cmi": round(cmi, 2),
        "switch_points": switch_points,
        "cmi_all": round(cmi_all, 2),
        "dominant_lang": dominant_lang,
        "lang_tokens_count": n_lang,
    }


def identify_languages(tokens: list[str]) -> dict:
    """
    Returns a dictionary containing:
      - 'results': list of dicts {"Token": token, "Language": lang}
      - 'stats': dict with counts, CMI, switch points, and ratios
      - 'cmi_details': raw numerical CMI measurements
    """
    results = []
    
    # In Hinglish, context is Hindi-English
    words = [t.lower() for t in tokens if WORD_RE.match(t)]
    lang_context = "hi"
    
    # Check if the sentence contains Devanagari characters
    has_devanagari = any('\u0900' <= char <= '\u097F' for t in tokens for char in t)
    
    counts = {"Hindi": 0, "English": 0, "Other": 0}

    for token in tokens:
        shape = token_shape(token)
        clean = token.strip(".,!?\"'-()[]{}")
        if shape in ("<PUNCT>", "<NUM>", "<URL>", "<MENTION>", "<HASHTAG>") or not clean:
            lang = "Other"
        else:
            if is_english(clean.lower(), lang_context):
                lang = "English"
            elif has_devanagari and clean.isascii() and clean.isalpha():
                # In a Devanagari-script sentence, pure ASCII alphabetic words are English
                lang = "English"
            else:
                lang = "Hindi"
                
        results.append({"Token": token, "Language": lang})
        counts[lang] += 1
        
    total_words = counts["Hindi"] + counts["English"]
    ratio = (counts["English"] / total_words * 100) if total_words > 0 else 0.0
        
    # Calculate CMI and Switch Points
    cmi_data = calculate_cmi(results)

    stats = {
        "Hindi Tokens": counts["Hindi"],
        "English Tokens": counts["English"],
        "Other Tokens": counts["Other"],
        "Total Tokens": len(tokens),
        "Language Words": cmi_data["lang_tokens_count"],
        "Code-Mixing Index (CMI)": f"{cmi_data['cmi']:.1f}%",
        "Switch Points": cmi_data["switch_points"],
        "CMI (Extended)": f"{cmi_data['cmi_all']:.1f}%",
        "Dominant Language": cmi_data["dominant_lang"],
        "Code-Mixing Ratio": f"{ratio:.1f}% (English words)",
    }
    
    return {
        "results": results, 
        "stats": stats,
        "cmi_details": cmi_data,
    }