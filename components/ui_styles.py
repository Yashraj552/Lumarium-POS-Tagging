"""
ui_styles.py — Design tokens, custom CSS, and responsive styling for Grammar Buddy.
Features a warm, friendly, minimalist aesthetic designed for students with soft pastels,
rounded cards, and progressive disclosure.
"""

TAG_METADATA = {
    "NOUN": {
        "name": "Naming Word",
        "desc": "Names a person, place, thing, or concept.",
        "bg": "#E0F2FE",      # Soft Sky Blue
        "border": "#BAE6FD",
        "text": "#0369A1",
        "badge_bg": "#0284C7",
        "emoji": "📦",
        "group": "Nouns"
    },
    "PROPN": {
        "name": "Special Name",
        "desc": "A specific name of a person, place, or brand.",
        "bg": "#DBEAFE",      # Soft Indigo/Blue
        "border": "#BFDBFE",
        "text": "#1D4ED8",
        "badge_bg": "#2563EB",
        "emoji": "🏷️",
        "group": "Nouns"
    },
    "VERB": {
        "name": "Action Word",
        "desc": "Tells what someone is doing or experiencing.",
        "bg": "#FEE2E2",      # Soft Coral/Rose
        "border": "#FECACA",
        "text": "#B91C1C",
        "badge_bg": "#DC2626",
        "emoji": "⚡",
        "group": "Verbs"
    },
    "AUX": {
        "name": "Helping Verb",
        "desc": "Helps the main verb show tense or mood (e.g. hai, tha, aahe, is, have).",
        "bg": "#FFE4E6",      # Soft Pink
        "border": "#FECDD3",
        "text": "#BE123C",
        "badge_bg": "#E11D48",
        "emoji": "🤝",
        "group": "Verbs"
    },
    "ADJ": {
        "name": "Describing Word",
        "desc": "Describes how a noun looks, feels, tastes, or behaves.",
        "bg": "#DCFCE7",      # Soft Mint Green
        "border": "#BBF7D0",
        "text": "#15803D",
        "badge_bg": "#16A34A",
        "emoji": "🎨",
        "group": "Adjectives"
    },
    "ADV": {
        "name": "How / When Word",
        "desc": "Tells how, when, where, or how much an action happened.",
        "bg": "#FEF3C7",      # Soft Warm Amber
        "border": "#FDE68A",
        "text": "#B45309",
        "badge_bg": "#D97706",
        "emoji": "⏱️",
        "group": "Adverbs"
    },
    "PRON": {
        "name": "Standing-in Word",
        "desc": "Replaces a noun (e.g. I, you, me, to, ti, maine, hum).",
        "bg": "#F3E8FF",      # Soft Lavender
        "border": "#E9D5FF",
        "text": "#7E22CE",
        "badge_bg": "#9333EA",
        "emoji": "👤",
        "group": "Pronouns"
    },
    "ADP": {
        "name": "Position / Connector",
        "desc": "Shows relations in time, place, or role (e.g. in, on, mein, la, ne).",
        "bg": "#F1F5F9",      # Soft Slate
        "border": "#E2E8F0",
        "text": "#334155",
        "badge_bg": "#475569",
        "emoji": "📍",
        "group": "Function words"
    },
    "DET": {
        "name": "Pointing Word",
        "desc": "Clarifies which noun is meant (e.g. the, a, this, ye, to).",
        "bg": "#F1F5F9",
        "border": "#E2E8F0",
        "text": "#334155",
        "badge_bg": "#475569",
        "emoji": "👉",
        "group": "Function words"
    },
    "PART": {
        "name": "Tiny Helper (Particle)",
        "desc": "Adds emphasis or negation (e.g. nahi, na, toh, pan, 'not').",
        "bg": "#F1F5F9",
        "border": "#E2E8F0",
        "text": "#334155",
        "badge_bg": "#475569",
        "emoji": "✨",
        "group": "Function words"
    },
    "CCONJ": {
        "name": "Sentence Linker",
        "desc": "Connects clauses or words (e.g. and, but, aur, aani, pan).",
        "bg": "#F1F5F9",
        "border": "#E2E8F0",
        "text": "#334155",
        "badge_bg": "#475569",
        "emoji": "🔗",
        "group": "Function words"
    },
    "SCONJ": {
        "name": "Sub-clause Linker",
        "desc": "Introduces a dependent condition or clause (e.g. because, if, agar).",
        "bg": "#F1F5F9",
        "border": "#E2E8F0",
        "text": "#334155",
        "badge_bg": "#475569",
        "emoji": "🪢",
        "group": "Function words"
    },
    "INTJ": {
        "name": "Emotion Exclamation",
        "desc": "Expresses sudden feelings (e.g. wow, arey, ouch, hey).",
        "bg": "#FDF2F8",      # Soft Rose/Pink
        "border": "#FCE7F3",
        "text": "#BE185D",
        "badge_bg": "#DB2777",
        "emoji": "🎉",
        "group": "Other"
    },
    "NUM": {
        "name": "Number Word",
        "desc": "Count or quantity (e.g. 1, 50, first, ek, do, don).",
        "bg": "#ECFDF5",      # Soft Sage Green
        "border": "#D1FAE5",
        "text": "#047857",
        "badge_bg": "#059669",
        "emoji": "🔢",
        "group": "Other"
    },
    "PUNCT": {
        "name": "Punctuation Mark",
        "desc": "Sentence punctuation (., !, ?, commas).",
        "bg": "#F8FAFC",
        "border": "#E2E8F0",
        "text": "#64748B",
        "badge_bg": "#94A3B8",
        "emoji": "🔣",
        "group": "Other"
    },
    "SYM": {
        "name": "Symbol",
        "desc": "Symbols like currency, %, emojis.",
        "bg": "#F8FAFC",
        "border": "#E2E8F0",
        "text": "#64748B",
        "badge_bg": "#94A3B8",
        "emoji": "🔤",
        "group": "Other"
    },
    "X": {
        "name": "Other Word",
        "desc": "Foreign words, unclassified tokens, or contractions.",
        "bg": "#F8FAFC",
        "border": "#E2E8F0",
        "text": "#64748B",
        "badge_bg": "#94A3B8",
        "emoji": "❓",
        "group": "Other"
    }
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1E293B;
}

h1, h2, h3, h4, .brand-font {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700;
}

/* App Background & Container */
.stApp {
    background-color: #F8FAFC;
}

.main-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0.5rem 1rem 3rem;
}

/* Hero Section */
.hero-box {
    background: linear-gradient(180deg, #FFFFFF 0%, #F1F5F9 100%);
    border: 1px solid #E2E8F0;
    border-radius: 24px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04);
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #64748B;
    max-width: 620px;
    margin: 0 auto 1.5rem;
    line-height: 1.6;
}

/* Feature Cards Grid */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.25rem;
    margin: 2rem 0;
}

.feature-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}

.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.05);
}

.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
    background: #F1F5F9;
    padding: 0.5rem;
    border-radius: 12px;
    display: inline-block;
}

.feature-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.35rem;
}

.feature-desc {
    font-size: 0.92rem;
    color: #64748B;
    line-height: 1.5;
}

/* Interactive Token Cards */
.token-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.85rem;
    margin: 1.25rem 0 2rem;
    justify-content: flex-start;
    align-items: stretch;
}

.token-card {
    border-radius: 16px;
    padding: 0.85rem 1.15rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    min-width: 95px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    cursor: pointer;
    position: relative;
    border-width: 2px;
    border-style: solid;
}

.token-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.08);
}

.token-card.active-token {
    box-shadow: 0 0 0 3px #38BDF8, 0 8px 16px -2px rgba(0, 0, 0, 0.1) !important;
}

.token-word {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.35rem;
}

.token-tag-badge {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    color: #FFFFFF;
    margin-bottom: 0.25rem;
}

.token-meaning {
    font-size: 0.78rem;
    font-weight: 600;
    opacity: 0.85;
}

/* Friendly Learning Summary */
.summary-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
}

.summary-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.summary-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.summary-badge-item {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    padding: 0.5rem 1rem;
    border-radius: 12px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #334155;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Script / Translation Trio Cards */
.translation-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin: 1.5rem 0;
}

.translation-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 1.4rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
}

.translation-label {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.translation-text {
    font-size: 1.25rem;
    font-weight: 600;
    color: #0F172A;
    line-height: 1.4;
}

/* Code Mixing Bar */
.mix-bar-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.mix-progress-track {
    height: 16px;
    border-radius: 10px;
    display: flex;
    overflow: hidden;
    margin: 0.85rem 0 1rem;
    background: #E2E8F0;
}

.mix-legend {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    font-size: 0.9rem;
    font-weight: 600;
}

.mix-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.35rem;
}

/* Popover / Token Inspection Box */
.inspector-box {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-left: 5px solid #0284C7;
    border-radius: 16px;
    padding: 1.35rem;
    margin: 1.25rem 0;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
}

.quiz-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
    margin: 1rem 0;
}

/* Grammar Reference Cards */
.guide-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}

.guide-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.05);
}

/* Streamlit button overrides for student warmth */
div.stButton > button:first-child {
    border-radius: 14px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    border: 1px solid #CBD5E1;
    transition: all 0.2s ease;
}

div.stButton > button:first-child:hover {
    border-color: #94A3B8;
    transform: translateY(-1px);
}

/* Primary CTA Button */
div.stButton > button[kind="primary"] {
    background: #0284C7;
    color: #FFFFFF;
    border: none;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
}

div.stButton > button[kind="primary"]:hover {
    background: #0369A1;
    box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35);
}

/* Modern Pill Styling for Navigation and Horizontal Radio Buttons */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.25rem 0;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 0.45rem 1.15rem;
    border-radius: 24px;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    font-weight: 600;
    font-size: 0.95rem;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    border-color: #38BDF8;
    background: #F0F9FF;
    transform: translateY(-1px);
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: #0284C7 !important;
    border-color: #0284C7 !important;
    color: #FFFFFF !important;
    font-weight: 700;
    box-shadow: 0 4px 10px rgba(2, 132, 199, 0.25);
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p,
div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] p {
    color: #FFFFFF !important;
}

/* Remove awkward top padding in Streamlit */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
</style>
"""
