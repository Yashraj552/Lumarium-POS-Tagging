"""
grammar_guide_data.py — Rich educational data for Grammar Buddy's interactive Grammar Guide.
Provides student-friendly explanations, Hinglish examples,
English examples, and memory hooks for all Universal Dependencies POS tags.
"""

GRAMMAR_CATEGORIES = {
    "All": None,
    "Nouns & Pronouns": ["NOUN", "PROPN", "PRON"],
    "Verbs & Actions": ["VERB", "AUX"],
    "Describing Words": ["ADJ", "ADV"],
    "Connectors & Helpers": ["ADP", "DET", "PART", "CCONJ", "SCONJ"],
    "Other": ["NUM", "INTJ", "PUNCT", "SYM", "X"]
}

GRAMMAR_GUIDE = [
    {
        "tag": "NOUN",
        "name": "Noun (Naming Word)",
        "emoji": "📦",
        "color": "#0284C7",
        "bg": "#E0F2FE",
        "definition": "A word that names a person, place, animal, object, or feeling.",
        "student_tip": "If you can see it, touch it, or name it, it's probably a noun!",
        "code_mixed_example": "Maine kal **school** mein **football** khela. / Mi **college** la jatoy.",
        "code_mixed_notes": "'school', 'football', and 'college' are all objects or places.",
        "english_example": "The **cat** chased the **ball** in the **garden**."
    },
    {
        "tag": "PROPN",
        "name": "Proper Noun (Special Name)",
        "emoji": "🏷️",
        "color": "#1D4ED8",
        "bg": "#DBEAFE",
        "definition": "A specific, unique name of an individual person, city, country, or brand. Usually starts with a capital letter.",
        "student_tip": "Look for names like your own name or city names!",
        "code_mixed_example": "**Rahul** ne **Mumbai** madhye **Google** join kela.",
        "code_mixed_notes": "'Rahul' (person), 'Mumbai' (place), and 'Google' (organization) are special proper names.",
        "english_example": "**Aarav** and **Priya** visited **London** during summer."
    },
    {
        "tag": "VERB",
        "name": "Verb (Action Word)",
        "emoji": "⚡",
        "color": "#DC2626",
        "bg": "#FEE2E2",
        "definition": "A word that expresses an action, process, or happening.",
        "student_tip": "Ask yourself: 'What is happening or being done?'",
        "code_mixed_example": "Tu homework **complete** **kiya**? / Mi gaana **aiktoy**.",
        "code_mixed_notes": "'complete kiya' (did complete) and 'aiktoy' (listening) represent the core actions.",
        "english_example": "We **ran** across the playground and **laughed**."
    },
    {
        "tag": "AUX",
        "name": "Auxiliary (Helping Verb)",
        "emoji": "🤝",
        "color": "#E11D48",
        "bg": "#FFE4E6",
        "definition": "A helping word that accompanies a main verb to show time (tense), permission, or ability.",
        "student_tip": "Words like 'is', 'was', 'hai', 'tha', 'aahe', 'hote' help the main action.",
        "code_mixed_example": "To yet **aahe**. / Train late ho **gayi** **hai**.",
        "code_mixed_notes": "'aahe' and 'hai' are helping verbs showing that the action is happening right now.",
        "english_example": "She **is** reading a book and **has** finished her homework."
    },
    {
        "tag": "PRON",
        "name": "Pronoun (Standing-in Word)",
        "emoji": "👤",
        "color": "#9333EA",
        "bg": "#F3E8FF",
        "definition": "A word used in place of a noun so you don't have to repeat names over and over.",
        "student_tip": "Words like 'I', 'you', 'me', 'maine', 'tu', 'to', 'ti', 'hum'.",
        "code_mixed_example": "**Maine** usko bola pan **to** reply nahi dila.",
        "code_mixed_notes": "'Maine' (I) and 'to' (he) stand in for actual people.",
        "english_example": "**She** told **me** that **they** won the quiz."
    },
    {
        "tag": "ADJ",
        "name": "Adjective (Describing Word)",
        "emoji": "🎨",
        "color": "#16A34A",
        "bg": "#DCFCE7",
        "definition": "A word that describes or gives more information about a noun (color, size, mood, quality).",
        "student_tip": "Ask: 'What kind of noun is this? How does it look or feel?'",
        "code_mixed_example": "Aaj weather khup **mast** aahe! / Ye movie bohot **boring** thi.",
        "code_mixed_notes": "'mast' (great) describes weather; 'boring' describes movie.",
        "english_example": "A **bright** rainbow appeared after the **heavy** rain."
    },
    {
        "tag": "ADV",
        "name": "Adverb (How / When / Where Word)",
        "emoji": "⏱️",
        "color": "#D97706",
        "bg": "#FEF3C7",
        "definition": "A word that gives extra details about a verb, adjective, or another adverb (telling when, how, or to what degree).",
        "student_tip": "Tells: 'When?', 'How?', or 'How much?'. (e.g. kal, aaj, bohot, quickly).",
        "code_mixed_example": "Maine **kal** message bheja tha, to **fast** chaltoy.",
        "code_mixed_notes": "'kal' tells when; 'fast' tells how.",
        "english_example": "He completed the assignment **yesterday** and answered **brilliantly**."
    },
    {
        "tag": "ADP",
        "name": "Adposition / Postposition (Position Word)",
        "emoji": "📍",
        "color": "#475569",
        "bg": "#F1F5F9",
        "definition": "Words (prepositions in English, postpositions/case markers in Hindi) showing relationships of place, direction, time, or role.",
        "student_tip": "Notice how English says 'in school' (preposition), but Hindi says 'school **mein**' or 'ghar **pe**' (postposition)!",
        "code_mixed_example": "School **mein** baitha tha / Mi Pune-**la** gelo.",
        "code_mixed_notes": "'mein' and '-la' show direction/location relation.",
        "english_example": "The key is **on** the desk **near** the window."
    },
    {
        "tag": "DET",
        "name": "Determiner (Pointing Word)",
        "emoji": "👉",
        "color": "#475569",
        "bg": "#F1F5F9",
        "definition": "A word that introduces a noun and specifies which one or how many (a, an, the, this, that, ye, vo).",
        "student_tip": "Points out which specific thing we are talking about.",
        "code_mixed_example": "**Ye** book meri hai, **the** principal called him.",
        "code_mixed_notes": "'Ye' and 'the' clarify specific items or persons.",
        "english_example": "**This** pencil belongs to **the** student."
    },
    {
        "tag": "PART",
        "name": "Particle (Tiny Helper / Emphasis)",
        "emoji": "✨",
        "color": "#475569",
        "bg": "#F1F5F9",
        "definition": "A small functional word that adds focus, emphasis, or negation without being a full adverb or verb (e.g. not, nahi, na, toh, pan).",
        "student_tip": "Includes words of negation like 'nahi' / 'not' and emphasis markers like 'toh'.",
        "code_mixed_example": "Maine bola **pan** to mana **nahi** kela.",
        "code_mixed_notes": "'nahi' negates the action; 'pan' serves as an emphasis/particle.",
        "english_example": "Please do **not** touch the wet paint."
    },
    {
        "tag": "CCONJ",
        "name": "Coordinating Conjunction (Sentence Linker)",
        "emoji": "🔗",
        "color": "#475569",
        "bg": "#F1F5F9",
        "definition": "Connects words, phrases, or independent thoughts equally (e.g. and, but, or, aur, aani, lekin).",
        "student_tip": "The bridge between two equal ideas!",
        "code_mixed_example": "Chai **aur** samosa kha lo / Cricket **aani** football aavadte.",
        "code_mixed_notes": "'aur' and 'aani' join two items directly.",
        "english_example": "I like science **and** music, **but** math is my favorite."
    },
    {
        "tag": "SCONJ",
        "name": "Subordinating Conjunction (Clause Linker)",
        "emoji": "🪢",
        "color": "#475569",
        "bg": "#F1F5F9",
        "definition": "Links a dependent clause to a main sentence (e.g. because, although, if, agar, karan).",
        "student_tip": "Introduces reasons, conditions, or causes.",
        "code_mixed_example": "Class miss zhali **karan** bus late hoti / **Agar** time mile toh aana.",
        "code_mixed_notes": "'karan' (because) explains the reason for missing class.",
        "english_example": "We stayed indoors **because** it began to thunderstorm."
    },
    {
        "tag": "NUM",
        "name": "Numeral (Number Word)",
        "emoji": "🔢",
        "color": "#059669",
        "bg": "#ECFDF5",
        "definition": "Any word or symbol denoting a quantity, count, or order.",
        "student_tip": "Numbers like 1, 2, 3, first, second, ek, don, tin.",
        "code_mixed_example": "Mere paas **2** tickets hain / **Pehla** period free tha.",
        "code_mixed_notes": "'2' and 'pehla' quantify or order items.",
        "english_example": "There are **seven** planets in this diagram."
    },
    {
        "tag": "INTJ",
        "name": "Interjection (Emotion Exclamation)",
        "emoji": "🎉",
        "color": "#DB2777",
        "bg": "#FDF2F8",
        "definition": "An exclamation showing sudden emotion, surprise, or greeting (e.g. wow, arey, ouch, yay, hello).",
        "student_tip": "Expresses excitement, surprise, or shock!",
        "code_mixed_example": "**Arey** wah! bohot badhiya score aaya.",
        "code_mixed_notes": "'Arey' and 'wah' express excitement.",
        "english_example": "**Wow**! That was an incredible performance."
    },
    {
        "tag": "PUNCT",
        "name": "Punctuation Mark",
        "emoji": "🔣",
        "color": "#64748B",
        "bg": "#F8FAFC",
        "definition": "Marks like periods, question marks, commas, and quotation marks that organize sentence structure.",
        "student_tip": "They tell you when to pause, stop, or ask a question.",
        "code_mixed_example": "Tu kuthun aalas **?**",
        "code_mixed_notes": "'?' marks a question.",
        "english_example": "Hello **,** world **!**"
    }
]
