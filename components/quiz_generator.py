"""
quiz_generator.py — Generates interactive educational grammar quiz questions
dynamically from the actual NLP output of the user's analyzed sentence.
"""
from __future__ import annotations

import random
from typing import Any
from components.ui_styles import TAG_METADATA


def generate_practice_questions(analysis_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Creates dynamic quiz questions strictly based on the real NLP pipeline outputs.
    Falls back to curated educational questions if the sentence is too short.
    """
    if not analysis_data or not analysis_data.get("pos_results"):
        return _fallback_questions()

    pos_results = analysis_data["pos_results"]
    lang_id_results = analysis_data.get("language_id", {}).get("results", [])
    entities = analysis_data.get("entities", {}).get("original", [])

    questions = []

    # Map words to tags and languages
    token_to_tag = {r["token"]: r["tag"] for r in pos_results}
    meaningful_tokens = [
        r["token"] for r in pos_results
        if r["tag"] not in ("PUNCT", "SYM", "X") and len(r["token"]) > 1
    ]

    # 1. Question: Find the Verb
    verbs = [r["token"] for r in pos_results if r["tag"] in ("VERB", "AUX") and r["token"].isalpha()]
    non_verbs = [r["token"] for r in pos_results if r["tag"] not in ("VERB", "AUX", "PUNCT", "SYM", "X") and r["token"].isalpha()]
    if verbs and non_verbs:
        target_verb = random.choice(verbs)
        distractors = random.sample(non_verbs, min(3, len(non_verbs)))
        options = list(set([target_verb] + distractors))
        random.shuffle(options)
        questions.append({
            "id": f"q_verb_{target_verb}",
            "type": "find_pos",
            "title": "⚡ Action Finder",
            "question": f"Which word in your sentence is the **Verb (Action Word)**?",
            "options": options,
            "correct": target_verb,
            "explanation_correct": f"🎉 Correct! **'{target_verb}'** is a verb because it describes an action, state, or event in your sentence.",
            "explanation_incorrect": f"Almost! A verb tells us what action is taking place or helps show time (tense). Keep an eye out for action words!",
        })

    # 2. Question: Find the Noun
    nouns = [r["token"] for r in pos_results if r["tag"] in ("NOUN", "PROPN") and r["token"].isalpha()]
    non_nouns = [r["token"] for r in pos_results if r["tag"] not in ("NOUN", "PROPN", "PUNCT", "SYM", "X") and r["token"].isalpha()]
    if nouns and non_nouns:
        target_noun = random.choice(nouns)
        distractors = random.sample(non_nouns, min(3, len(non_nouns)))
        options = list(set([target_noun] + distractors))
        random.shuffle(options)
        questions.append({
            "id": f"q_noun_{target_noun}",
            "type": "find_pos",
            "title": "📦 Naming Word Finder",
            "question": f"Which word in your sentence is a **Noun (Naming Word)**?",
            "options": options,
            "correct": target_noun,
            "explanation_correct": f"🎉 Spot on! **'{target_noun}'** is a noun because it names a person, place, thing, or concept.",
            "explanation_incorrect": "Not quite! Remember: nouns are words that name things you can point to, places, or people.",
        })

    # 3. Question: Identify POS Tag for a specific word
    if meaningful_tokens:
        target_token = random.choice(meaningful_tokens)
        correct_tag = token_to_tag[target_token]
        tag_meta = TAG_METADATA.get(correct_tag, TAG_METADATA["NOUN"])
        friendly_correct = f"{correct_tag} ({tag_meta['name']})"

        other_tags = [t for t in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP"] if t != correct_tag]
        distractor_tags = random.sample(other_tags, min(3, len(other_tags)))
        friendly_options = [friendly_correct] + [
            f"{t} ({TAG_METADATA.get(t, {}).get('name', t)})" for t in distractor_tags
        ]
        random.shuffle(friendly_options)

        questions.append({
            "id": f"q_tag_{target_token}",
            "type": "identify_tag",
            "title": "🏷️ POS Tag Detective",
            "question": f"What part of speech is the word **'{target_token}'** in this sentence?",
            "options": friendly_options,
            "correct": friendly_correct,
            "explanation_correct": f"🎉 Excellent! **'{target_token}'** functions as **{correct_tag}** ({tag_meta['name']}): {tag_meta['desc']}",
            "explanation_incorrect": f"Give it another thought! {tag_meta['desc']}",
        })

    # 4. Question: Identify the English word (Code-Mixing Quiz)
    english_words = [r["Token"] for r in lang_id_results if r["Language"] == "English" and len(r["Token"]) > 1]
    regional_words = [r["Token"] for r in lang_id_results if r["Language"] in ("Hindi", "Marathi") and len(r["Token"]) > 1]
    if english_words and regional_words:
        target_eng = random.choice(english_words)
        distractors = random.sample(regional_words, min(3, len(regional_words)))
        options = list(set([target_eng] + distractors))
        random.shuffle(options)
        questions.append({
            "id": f"q_eng_{target_eng}",
            "type": "code_mix",
            "title": "🔀 Language Detective",
            "question": "Which of these words from your sentence comes from **English**?",
            "options": options,
            "correct": target_eng,
            "explanation_correct": f"🎉 Exactly! **'{target_eng}'** is an English word used naturally inside your sentence.",
            "explanation_incorrect": "Look closely! That word comes from Hindi. Which one is borrowed from English?",
        })

    # 5. Question: Find the Pronoun
    pronouns = [r["token"] for r in pos_results if r["tag"] == "PRON" and r["token"].isalpha()]
    if pronouns and len(pos_results) > 2:
        target_pron = random.choice(pronouns)
        non_prons = [r["token"] for r in pos_results if r["token"] != target_pron and len(r["token"]) > 1 and r["token"].isalpha()]
        distractors = random.sample(non_prons, min(3, len(non_prons)))
        options = list(set([target_pron] + distractors))
        random.shuffle(options)
        questions.append({
            "id": f"q_pron_{target_pron}",
            "type": "find_pos",
            "title": "👤 Pronoun Finder",
            "question": "Which word is a **Pronoun (Standing-in Word)** in your sentence?",
            "options": options,
            "correct": target_pron,
            "explanation_correct": f"🎉 Well done! **'{target_pron}'** is a pronoun standing in for a person or subject.",
            "explanation_incorrect": "A pronoun replaces someone's name (like 'maine', 'tu', 'he', 'we'). Try again!",
        })

    # 6. Question: Named Entity (if detected)
    if entities:
        target_ent = random.choice(entities)
        ent_text = target_ent.get("Entity", "")
        ent_type = target_ent.get("Entity Type", "")
        if ent_text:
            other_types = [t for t in ["PERSON", "LOCATION", "ORGANIZATION", "DATE"] if t != ent_type]
            options = [ent_type] + random.sample(other_types, min(3, len(other_types)))
            random.shuffle(options)
            questions.append({
                "id": f"q_ent_{ent_text}",
                "type": "ner",
                "title": "🔍 Named Entity Recognition",
                "question": f"What kind of entity is **'{ent_text}'**?",
                "options": options,
                "correct": ent_type,
                "explanation_correct": f"🎉 Perfect! **'{ent_text}'** was identified as a **{ent_type}**.",
                "explanation_incorrect": f"Think about whether '{ent_text}' represents a person, a place, an organization, or a time.",
            })

    return questions if questions else _fallback_questions()


def _fallback_questions() -> list[dict[str, Any]]:
    """Curated questions used when no sentence has been analyzed yet."""
    return [
        {
            "id": "q_default_1",
            "type": "find_pos",
            "title": "⚡ Action Finder",
            "question": "In the sentence: *'Maine kal school mein football khela'*, which word is the **Verb**?",
            "options": ["Maine", "school", "football", "khela"],
            "correct": "khela",
            "explanation_correct": "🎉 Correct! 'khela' (played) describes the action that took place.",
            "explanation_incorrect": "Look for the word that tells what action was performed!",
        },
        {
            "id": "q_default_2",
            "type": "identify_tag",
            "title": "🏷️ POS Tag Detective",
            "question": "In: *'Tu homework complete kiya?'*, what part of speech is **'homework'**?",
            "options": ["NOUN (Naming Word)", "VERB (Action Word)", "ADJ (Describing Word)", "ADV (How / When Word)"],
            "correct": "NOUN (Naming Word)",
            "explanation_correct": "🎉 Correct! 'homework' is an English noun representing the task assigned.",
            "explanation_incorrect": "'homework' is an object or assignment, which makes it a noun!",
        },
        {
            "id": "q_default_3",
            "type": "code_mix",
            "title": "🔀 Language Detective",
            "question": "In: *'Main college ja raha hoon'*, which word is borrowed from **English**?",
            "options": ["Main", "college", "ja", "raha"],
            "correct": "college",
            "explanation_correct": "🎉 Exactly! 'college' is an English word code-mixed into a Hindi sentence.",
            "explanation_incorrect": "Find the word that is originally from the English vocabulary!",
        }
    ]
