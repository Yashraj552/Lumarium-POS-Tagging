"""
tagset.py — the ONE tag set every module speaks.

Every tagger (rules, HMM, transformer) and every dataset passes its labels
through `normalize_tag()` before anything is compared or scored. Without this,
"NOUN" from the transformer and "N_NN" from the dataset would count as a
disagreement / an error even when they mean the same thing.

We use Universal Dependencies (UD) POS tags, which is what the LinCE
Hindi-English benchmark (and most recent code-mixed corpora) use.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

UD_TAGS: tuple[str, ...] = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)
_UD_SET = frozenset(UD_TAGS)

# Known non-UD spellings -> UD. Extend this if your dataset uses another scheme.
TAG_ALIASES: dict[str, str] = {
    # UD v1 / common variants
    "CONJ": "CCONJ", "PRT": "PART", "PRN": "PRON", "NNP": "PROPN", ".": "PUNCT",
    "PUNC": "PUNCT", "NUMBER": "NUM", "INTERJ": "INTJ", "UNK": "X", "O": "X",
    # Penn Treebank (English side of some corpora)
    "NN": "NOUN", "NNS": "NOUN", "NNPS": "PROPN", "VB": "VERB", "VBD": "VERB",
    "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB", "MD": "AUX",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ", "RB": "ADV", "RBR": "ADV",
    "RBS": "ADV", "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "DT": "DET",
    "IN": "ADP", "CC": "CCONJ", "CD": "NUM", "UH": "INTJ", "RP": "PART",
    "TO": "PART", "FW": "X", "SYM": "SYM",
    # ICON-2016 / Indian-language shared-task style (G_ = generic, N_ = noun...)
    "G_N": "NOUN", "N_NN": "NOUN", "N_NNP": "PROPN", "N_NST": "NOUN",
    "G_V": "VERB", "V_VM": "VERB", "V_VAUX": "AUX", "G_J": "ADJ", "JJ_": "ADJ",
    "G_R": "ADV", "RB_": "ADV", "G_PRP": "PRON", "PR_PRP": "PRON",
    "PR_PRQ": "PRON", "PR_PRF": "PRON", "PR_PRL": "PRON", "PR_PRC": "PRON",
    "DM_DMD": "DET", "DM_DMQ": "DET", "DM_DMR": "DET", "DM_DMI": "DET",
    "G_SYM": "SYM", "RD_PUNC": "PUNCT", "RD_SYM": "SYM", "RD_RDF": "X",
    "RD_UNK": "X", "RD_ECH": "X", "CC_CCD": "CCONJ", "CC_CCS": "SCONJ",
    "PSP": "ADP", "RP_RPD": "PART", "RP_NEG": "PART", "RP_INTF": "ADV",
    "RP_INJ": "INTJ", "QT_QTC": "NUM", "QT_QTF": "DET", "QT_QTO": "ADJ",
    # LinCE specific / UD extensions
    "PART_NEG": "PART", "PRON_WH": "PRON",
    "G_X": "X", "E": "X", "U": "X", "@": "X", "#": "X", "~": "X",
    "$": "NUM", ",": "PUNCT", "&": "CCONJ",
}

_warned: set[str] = set()


def normalize_tag(tag: str | None) -> str:
    """Map any incoming label to a UD tag. Unknown labels become 'X' (warned once)."""
    if tag is None:
        return "X"
    raw = str(tag).strip()
    # HF models sometimes emit BIO-style labels ("B-NOUN") — strip the prefix.
    if len(raw) > 2 and raw[1] == "-" and raw[0] in "BIES":
        raw = raw[2:]
    upper = raw.upper()
    if upper in _UD_SET:
        return upper
    if upper in TAG_ALIASES:
        return TAG_ALIASES[upper]
    if raw in TAG_ALIASES:
        return TAG_ALIASES[raw]
    if upper not in _warned:
        _warned.add(upper)
        logger.warning("Unknown tag %r mapped to 'X' — add it to TAG_ALIASES.", tag)
    return "X"
