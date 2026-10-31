"""
text_utils.py — tokenization and normalization shared by every module.

Keeping these in one place guarantees the rule tagger, HMM and UI all see a
token the same way (e.g. "Sooooo" is normalized identically everywhere).
"""
from __future__ import annotations

import re
import unicodedata

# Order matters: the first alternative that matches wins.
_TOKEN_RE = re.compile(
    r"""
    https?://\S+ | www\.\S+                 # URLs
    | [@#][\w_]+                            # @mentions and #hashtags
    | \d+(?:[.,:/]\d+)*(?:st|nd|rd|th)?     # numbers, times, dates, 1st/2nd
    | [^\W\d_]+(?:[-'][^\W\d_]+)*           # words, incl. dog-ne, don't
    | [^\w\s]                               # any single punctuation / emoji
    """,
    re.VERBOSE | re.UNICODE,
)

_REPEAT_3PLUS = re.compile(r"(.)\1{2,}")
_REPEAT_2PLUS = re.compile(r"(.)\1+")

URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.I)
MENTION_RE = re.compile(r"^@\w+$")
HASHTAG_RE = re.compile(r"^#\w+$")
NUMBER_RE = re.compile(r"^\d+(?:[.,:/]\d+)*(?:st|nd|rd|th)?$", re.I)


def tokenize(text: str) -> list[str]:
    """Split raw text into tokens, keeping hyphenated/fused forms like 'dog-ne' whole."""
    return _TOKEN_RE.findall(text)


_COMMON_SINGLE_ENDINGS = frozenset({
    "so", "no", "to", "go", "do", "lo", "yo", "ho", "bhai", "hai", "plz", "yes",
    "bahut", "yaar", "kyu", "kya", "na", "haan", "acha", "achha", "karo", "raha",
    "rahi", "rahe", "hoga", "hogi", "hoge", "tha", "thi", "the",
})


def clean_elongation(token: str) -> str:
    """
    Intelligently collapses elongations:
      - 'sooooo' -> 'so' (matches single-char ending root)
      - 'cooool' -> 'cool' (preserves valid double letters)
    """
    squashed = _REPEAT_2PLUS.sub(r"\1", token.lower())
    if squashed in _COMMON_SINGLE_ENDINGS:
        return squashed
    return _REPEAT_3PLUS.sub(r"\1\1", token.lower())


def normalize_token(token: str) -> str:
    """
    Lowercase and collapse elongations: 'Sooooo' -> 'soo', 'cooool' -> 'cool'.

    Runs of 3+ identical characters are cut to 2 (not 1), because English has
    real double letters ('cool', 'good') but essentially no triples.
    Use `squash_token` for the aggressive single-letter version.
    """
    return _REPEAT_3PLUS.sub(r"\1\1", token.lower())


def squash_token(token: str) -> str:
    """Aggressive variant: collapse every repeat to one char ('soo' -> 'so')."""
    return _REPEAT_2PLUS.sub(r"\1", token.lower())


def is_punct(token: str) -> bool:
    """True if every character is punctuation or a symbol (incl. emoji)."""
    return bool(token) and all(unicodedata.category(c)[0] in "PS" for c in token)


def token_shape(token: str) -> str | None:
    """
    Return a coarse class for tokens whose *form* matters more than the word
    itself, or None for ordinary words. The HMM uses these as pseudo-words so
    that every URL / number / emoji shares statistics.
    """
    if URL_RE.match(token):
        return "<URL>"
    if MENTION_RE.match(token):
        return "<MENTION>"
    if HASHTAG_RE.match(token):
        return "<HASHTAG>"
    if NUMBER_RE.match(token):
        return "<NUM>"
    if is_punct(token):
        return "<PUNCT>"
    return None
