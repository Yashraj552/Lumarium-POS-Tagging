"""
Tests for new NLP features (Morphology, Language ID, NER).
"""
import pytest
from morphology import analyze_morphology
from language_identifier import identify_languages
from ner import extract_entities
from english_validation import validate_english_pos
from rule_tagger import RuleBasedTagger

def test_morphology():
    tokens = ["ladkon", "dog-ne", "kiya"]
    # Dummy pos results just to simulate context
    pos_results = [{"token": "ladkon", "tag": "NOUN"}, 
                   {"token": "dog-ne", "tag": "NOUN"},
                   {"token": "kiya", "tag": "VERB"}]
    
    res = analyze_morphology(tokens, pos_results)
    
    assert len(res) == 3
    # Check plural logic for Hindi nouns
    assert res[0]["Token"] == "ladkon"
    assert res[0]["Number"] == "Plural"
    
    # Check hyphenated case logic
    assert res[1]["Token"] == "dog-ne"
    assert res[1]["Root"] == "dog"
    assert res[1]["Suffix"] == "-ne"
    assert res[1]["Case"] == "Oblique/Marker"
    
    # Check light verb
    assert res[2]["Token"] == "kiya"
    assert res[2]["Other"] == "Light Verb + Past Tense"

def test_language_identifier():
    tokens = ["Maine", "aaj", "college", "mein", "assignment", "complete", "kiya", "."]
    res = identify_languages(tokens)
    
    assert res["stats"]["Total Tokens"] == 8
    
    langs = {r["Token"]: r["Language"] for r in res["results"]}
    assert langs["Maine"] == "Hindi"
    assert langs["aaj"] == "Hindi"
    assert langs["college"] == "English"
    assert langs["mein"] == "Hindi"
    assert langs["assignment"] == "English"
    assert langs["complete"] == "English"
    assert langs["kiya"] == "Hindi"
    assert langs["."] == "Other"

def test_ner():
    tokens = ["Main", "kal", "mumbai", "mein", "tcs", "gaya", "."]
    entities = extract_entities(tokens)
    
    assert len(entities) >= 2
    
    locs = [e["Entity"] for e in entities if e["Entity Type"] == "LOCATION"]
    orgs = [e["Entity"] for e in entities if e["Entity Type"] == "ORGANIZATION"]
    
    assert "mumbai" in locs
    assert "tcs" in orgs

def test_english_validation():
    ensemble_results = [
        {"token": "morning", "tag": "VERB", "source": "hmm"},
        {"token": "sun", "tag": "NOUN", "source": "hmm"}
    ]
    res = validate_english_pos(ensemble_results)
    
    assert len(res) == 2
    # 'morning' as VERB should be contradicted by WordNet NOUN evidence
    assert res[0]["Token"] == "morning"
    assert res[0]["Ensemble POS"] == "VERB"
    assert "NOUN" in res[0]["WordNet Evidence"]
    # Expect Validated POS to be corrected (probably to NOUN)
    assert res[0]["Validated POS"] != "VERB"
    assert res[0]["Validated POS"] == "NOUN"
    
    # 'sun' as NOUN should be accepted
    assert res[1]["Token"] == "sun"
    assert res[1]["Ensemble POS"] == "NOUN"
    assert res[1]["Validated POS"] == "NOUN"
