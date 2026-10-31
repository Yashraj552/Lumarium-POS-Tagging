"""
ner.py — New Feature 3: Named Entity Recognition

A lightweight NER implementation specifically for code-mixed text.
Avoids heavy dependencies like spaCy by using heuristics, regex, and gazetteers.
Detects PERSON, LOCATION, ORGANIZATION, DATE, TIME, and MONEY.
"""
from __future__ import annotations

import re

# Gazetteers for a lightweight NER without model dependencies
LOCATIONS = {"mumbai", "pune", "delhi", "bangalore", "india", "america", "london", 
             "मुंबई", "पुणे", "दिल्ली"}
ORGANIZATIONS = {"tcs", "infosys", "wipro", "google", "microsoft", "amazon", "facebook"}
PERSON_NAMES = {"rahul", "sachin", "amit", "riya", "priya", "rohit", "virat",
                "राहुल", "सचिन", "अमित"}
DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "somwar", "mangalwar", "budhwar", "guruwar", "shukrawar", "shaniwar", "raviwar", "itwar",
        "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार", "इतवार",
        "today", "tomorrow", "yesterday", "aaj", "kal", "parso", "आज", "कल", "परसों"}
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
          "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}

# Regex patterns
TIME_RE = re.compile(r"^(?:(?:[0-1]?[0-9]|2[0-3]):[0-5][0-9](?:\s?[aApP][mM])?)|(?:[0-9]{1,2}\s?[aApP][mM])$")
DATE_RE = re.compile(r"^(?:[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})|(?:[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})$")
MONEY_RE = re.compile(r"^(?:Rs\.?|\$|₹|€|£)?\s?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?\s?(?:rs|rupees|dollars|bucks)?$", re.IGNORECASE)

def extract_entities(tokens: list[str]) -> list[dict]:
    """
    Returns a list of detected named entities.
    Example output format: [{"Entity": "TCS", "Entity Type": "ORGANIZATION"}]
    """
    entities = []
    
    for token in tokens:
        lower = token.lower()
        entity_type = None
        
        # Strip some common punctuation for checking
        clean_token = token.strip(",.!?\"'")
        clean_lower = clean_token.lower()
        
        if not clean_token:
            continue
            
        if clean_lower in LOCATIONS:
            entity_type = "LOCATION"
        elif clean_lower in ORGANIZATIONS:
            entity_type = "ORGANIZATION"
        elif clean_lower in PERSON_NAMES:
            entity_type = "PERSON"
        elif clean_lower in DAYS or clean_lower in MONTHS:
            entity_type = "DATE"
        elif TIME_RE.match(clean_token) or clean_lower in {"morning", "evening", "night", "subah", "shaam", "dopahar", "raat", "सुबह", "शाम", "दोपहर", "रात"}:
            entity_type = "TIME" if TIME_RE.match(clean_token) else "TIME"
        elif DATE_RE.match(clean_token):
            entity_type = "DATE"
        elif MONEY_RE.match(clean_token) and any(char.isdigit() for char in clean_token):
            entity_type = "MONEY"
        # Capitalization heuristic (if not the first word and capitalized, and alpha)
        # We don't have sentence position here easily, so we just check Title Case 
        # but skip common start-of-sentence words if we can't be sure.
        elif clean_token.istitle() and len(clean_token) > 2 and clean_token.isascii():
            # A very weak heuristic for English names/entities not in gazetteer.
            # We won't tag it to avoid false positives, or we could tag as MISC
            pass
            
        if entity_type:
            # Check if we already have this entity to avoid duplicates (optional)
            entities.append({"Entity": clean_token, "Entity Type": entity_type})
            
    return entities
