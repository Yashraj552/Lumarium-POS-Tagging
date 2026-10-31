"""
app.py — Grammar Buddy: An Interactive Grammar Learning Platform for Students.
Redesigned with a warm, friendly, minimalist aesthetic while preserving 100% of the
underlying NLP research pipeline (Rules + HMM + Transformer + Multi-tier Arbiter).
"""
from __future__ import annotations

import html
from typing import Any
import pandas as pd
import streamlit as st

from ensemble import ArbiterConfig
from components.ui_styles import CUSTOM_CSS, TAG_METADATA
from components.grammar_guide_data import GRAMMAR_CATEGORIES, GRAMMAR_GUIDE
from components.quiz_generator import generate_practice_questions
from components.nlp_bridge import get_model_inventory, process_sentence

# ---------------------------------------------------------------- Page Config
st.set_page_config(
    page_title="Grammar Buddy 🌱",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject friendly design tokens & typography
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- Session State
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🏠 Home"
if "sentence" not in st.session_state:
    st.session_state.sentence = "Maine kal school mein football khela."
if "selected_token_idx" not in st.session_state:
    st.session_state.selected_token_idx = 0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_attempts" not in st.session_state:
    st.session_state.quiz_attempts = 0
if "current_q_idx" not in st.session_state:
    st.session_state.current_q_idx = 0
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "user_selected_ans" not in st.session_state:
    st.session_state.user_selected_ans = None

# Arbiter Configuration State
if "use_transformer" not in st.session_state:
    st.session_state.use_transformer = True
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.85
if "fallback" not in st.session_state:
    st.session_state.fallback = "hmm"
if "rule_overridable" not in st.session_state:
    st.session_state.rule_overridable = True

# ---------------------------------------------------------------- Model Caching
@st.cache_resource(show_spinner="Preparing Grammar Buddy's language models…")
def load_all_models(use_trf: bool):
    return get_model_inventory(use_transformer=use_trf)

inventory = load_all_models(st.session_state.use_transformer)

# Synchronize arbiter defaults on initial run
if "cfg_initialized" not in st.session_state:
    st.session_state.threshold = float(inventory.saved_cfg.threshold)
    st.session_state.fallback = inventory.saved_cfg.fallback
    st.session_state.rule_overridable = inventory.saved_cfg.rule_overridable
    st.session_state.cfg_initialized = True

current_arbiter_cfg = ArbiterConfig(
    threshold=st.session_state.threshold,
    fallback=st.session_state.fallback,
    rule_overridable=st.session_state.rule_overridable,
)


def run_analysis(sentence: str) -> dict[str, Any]:
    """Execute pipeline with caching."""
    return process_sentence(sentence, current_arbiter_cfg, inventory)


# Cached sentence analysis
analysis = run_analysis(st.session_state.sentence) if st.session_state.sentence.strip() else {}

# ---------------------------------------------------------------- Navigation Bar
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; padding: 0.5rem 0 1.25rem; border-bottom: 1px solid #E2E8F0; margin-bottom: 1.5rem;">
    <div style="display:flex; align-items:center; gap:0.6rem;">
        <span style="font-size:1.85rem;">🌱</span>
        <div>
            <div style="font-family:'Outfit', sans-serif; font-size:1.45rem; font-weight:800; color:#0F172A; letter-spacing:-0.01em;">Grammar Buddy</div>
            <div style="font-size:0.8rem; color:#64748B; font-weight:600;">Learn grammar from the way you speak</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_tabs = ["🏠 Home", "🔬 Analyze", "🎮 Practice", "📖 Grammar Guide", "🧪 NLP Lab"]

# Render navigation as sleek pill radio buttons
selected_nav = st.radio(
    "Navigation",
    nav_tabs,
    index=nav_tabs.index(st.session_state.active_tab) if st.session_state.active_tab in nav_tabs else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
)

if selected_nav and selected_nav != st.session_state.active_tab:
    st.session_state.active_tab = selected_nav
    st.rerun()

current_tab = st.session_state.active_tab

# ================================================================
# TAB 1: HOME PAGE
# ================================================================
if current_tab == "🏠 Home":
    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">Let's Learn Grammar! 🌱</div>
        <div class="hero-subtitle">
            Write a sentence the way you normally speak with friends or family, and we'll show you how every word works.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ✍️ Try your own sentence:")
    home_input = st.text_input(
        "Enter sentence",
        value=st.session_state.sentence,
        placeholder="e.g. Maine kal school mein football khela.",
        label_visibility="collapsed",
        key="home_sentence_input",
    )

    col_btn, col_gap = st.columns([1, 4])
    with col_btn:
        if st.button("Analyze Sentence ✨", type="primary", use_container_width=True):
            if home_input.strip():
                st.session_state.sentence = home_input.strip()
                st.session_state.selected_token_idx = 0
                st.session_state.active_tab = "🔬 Analyze"
                st.rerun()

    st.markdown("<p style='font-size:0.88rem; font-weight:600; color:#64748B; margin-top:0.75rem;'>Or click a fun example to explore:</p>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    examples = [
        "Maine kal school mein football khela.",
        "Tu homework complete kiya?",
        "Mi college la jatoy.",
        "Aaj weather khup mast aahe!"
    ]
    for i, ex in enumerate(examples):
        with chip_cols[i]:
            if st.button(f"💬 \"{ex[:22]}...\"", key=f"chip_home_{i}", use_container_width=True):
                st.session_state.sentence = ex
                st.session_state.selected_token_idx = 0
                st.session_state.active_tab = "🔬 Analyze"
                st.rerun()

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🏷️</div>
            <div class="feature-title">POS Tags</div>
            <div class="feature-desc">Discover what every word is doing in your sentence — naming, acting, or describing.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔀</div>
            <div class="feature-title">Code Mixing</div>
            <div class="feature-desc">See which words come from Hindi or English and how they blend together.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌱</div>
            <div class="feature-title">Word Structure</div>
            <div class="feature-desc">Understand roots, prefixes, suffixes, cases, and verb tenses in your natural speech.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Named Entities</div>
            <div class="feature-desc">Spot people, places, organizations, dates, and times mentioned in your text.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# TAB 2: ANALYZE PAGE
# ================================================================
elif current_tab == "🔬 Analyze":
    st.markdown("### 🔬 Sentence Explorer")
    st.markdown("<p style='color:#64748B; font-size:0.95rem; margin-top:-0.5rem;'>Type any Hinglish sentence below to inspect every word.</p>", unsafe_allow_html=True)

    input_cols = st.columns([5, 1])
    with input_cols[0]:
        new_sentence = st.text_input(
            "Sentence to analyze",
            value=st.session_state.sentence,
            placeholder="Type Hinglish text...",
            label_visibility="collapsed",
            key="analyze_sentence_input"
        )
    with input_cols[1]:
        if st.button("Analyze", type="primary", use_container_width=True):
            if new_sentence.strip():
                st.session_state.sentence = new_sentence.strip()
                st.session_state.selected_token_idx = 0
                st.rerun()

    # Re-run pipeline if sentence changed
    if new_sentence.strip() and new_sentence.strip() != st.session_state.sentence:
        st.session_state.sentence = new_sentence.strip()
        st.session_state.selected_token_idx = 0
        st.rerun()

    if not analysis or not analysis.get("pos_results"):
        st.info("💡 Type a sentence above to explore its grammar!")
    else:
        results = analysis["pos_results"]
        tokens = analysis["tokens"]
        curr_idx = min(st.session_state.selected_token_idx, len(results) - 1)

        # ---------------- 1. Interactive Word Cards ----------------
        st.markdown("<h4 style='margin-top:1.25rem;'>🏷️ Word Cards (Parts of Speech)</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.88rem; color:#64748B; margin-top:-0.3rem;'>Click any word below to see its friendly meaning and what the models learned:</p>", unsafe_allow_html=True)

        token_card_cols = st.columns(len(results))
        for idx, r in enumerate(results):
            meta = TAG_METADATA.get(r["tag"], TAG_METADATA["X"])
            is_active = (idx == curr_idx)
            with token_card_cols[idx]:
                btn_label = f"{r['token']}\n{r['tag']}"
                # Active card gets styled visual indicator
                active_star = " ⭐" if is_active else ""
                card_html = f"""
                <div style="background:{meta['bg']}; border:2px solid {meta['badge_bg'] if is_active else meta['border']};
                            border-radius:14px; padding:0.5rem 0.4rem; text-align:center; margin-bottom:0.4rem;
                            box-shadow:{'0 0 0 2px ' + meta['badge_bg'] if is_active else '0 2px 4px rgba(0,0,0,0.03)'};">
                    <div style="font-weight:700; font-size:1.05rem; color:#0F172A;">{html.escape(r['token'])}</div>
                    <div style="background:{meta['badge_bg']}; color:#FFFFFF; font-size:0.68rem; font-weight:700;
                                border-radius:12px; padding:0.15rem 0.4rem; display:inline-block; margin-top:0.25rem;">{r['tag']}</div>
                    <div style="font-size:0.7rem; color:{meta['text']}; font-weight:600; margin-top:0.2rem;">{meta['name']}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button(f"Inspect{active_star}", key=f"sel_tok_{idx}", use_container_width=True):
                    st.session_state.selected_token_idx = idx
                    st.rerun()

        # ---------------- 2. Token Details Inspector ----------------
        active_r = results[curr_idx]
        active_meta = TAG_METADATA.get(active_r["tag"], TAG_METADATA["X"])

        inspector_html = f"""
        <div class="inspector-box" style="border-left-color:{active_meta['badge_bg']};">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <span style="font-size:0.8rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.04em;">Selected Word</span>
                    <div style="font-size:1.8rem; font-weight:800; color:#0F172A; font-family:'Outfit', sans-serif;">{html.escape(active_r['token'])}</div>
                </div>
                <div style="text-align:right;">
                    <span style="background:{active_meta['badge_bg']}; color:#FFFFFF; padding:0.3rem 0.8rem; border-radius:20px; font-weight:700; font-size:0.9rem;">
                        {active_r['tag']} · {active_meta['name']}
                    </span>
                    <div style="font-size:0.82rem; color:#64748B; font-weight:600; margin-top:0.35rem;">Word class: {active_meta['group']}</div>
                </div>
            </div>
            <div style="background:#F8FAFC; border-radius:12px; padding:0.85rem 1rem; margin:1rem 0; font-size:0.95rem; color:#334155; line-height:1.5;">
                <strong>💡 Friendly Meaning:</strong> {active_meta['desc']}
            </div>
        """
        st.markdown(inspector_html, unsafe_allow_html=True)

        # Simplified "Why did the system choose this?"
        st.markdown("**🤔 Why did Grammar Buddy choose this tag?**")
        reason_summary = active_r.get("reason", "Determined by language models")
        
        # Build student-friendly status badges
        m_cols = st.columns(3)
        trf_cand = active_r["candidates"].get("transformer")
        rule_cand = active_r["candidates"].get("rule")
        hmm_cand = active_r["candidates"].get("hmm")

        with m_cols[0]:
            if trf_cand:
                conf_pct = int(trf_cand["conf"] * 100) if trf_cand["conf"] else 0
                st.info(f"🧠 **Transformer:** {trf_cand['tag']} ({conf_pct}% confidence)")
            else:
                st.caption("🧠 Transformer: Not engaged")

        with m_cols[1]:
            if rule_cand:
                st.info(f"📚 **Rule System:** {rule_cand['tag']} (Rule '{rule_cand.get('rule', '')}')")
            else:
                st.caption("📚 Rule System: No rule fired")

        with m_cols[2]:
            if hmm_cand:
                hmm_pct = int(hmm_cand["conf"] * 100) if hmm_cand["conf"] else 0
                st.info(f"📊 **HMM Model:** {hmm_cand['tag']} ({hmm_pct}% context score)")
            else:
                st.caption("📊 HMM Model: Not engaged")

        with st.expander("🔍 Show technical details for this word (NLP Lab view)"):
            st.write(f"**Final Tag:** `{active_r['tag']}` | **Source Tier:** `{active_r['source']}`")
            st.write(f"**Confidence Score:** `{active_r['conf']}`")
            st.write(f"**Arbiter Decision Rationale:** {reason_summary}")
            st.json(active_r["candidates"])

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- 3. Learning Summary ----------------
        tag_counts = analysis["tag_counts"]
        summary_badges_html = "".join([
            f"""<div class="summary-badge-item">
                <span style="font-size:1.1rem;">{TAG_METADATA.get(t, TAG_METADATA['X'])['emoji']}</span>
                <span><strong>{cnt}</strong> {TAG_METADATA.get(t, TAG_METADATA['X'])['name']} ({t})</span>
            </div>"""
            for t, cnt in tag_counts.items()
        ])

        st.markdown(f"""
        <div class="summary-container">
            <div class="summary-title">✨ What did we learn?</div>
            <div style="font-size:0.92rem; color:#475569; margin-bottom:0.75rem;">
                Your sentence has <strong>{len(tokens)} words</strong> with the following grammatical ingredients:
            </div>
            <div class="summary-badges">{summary_badges_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- 4. Translation & Script View ----------------
        trans_info = analysis.get("translation", {})
        if trans_info.get("available"):
            st.markdown("#### 🌐 Script & Translation View")
            st.markdown(f"""
            <div class="translation-grid">
                <div class="translation-card">
                    <div class="translation-label">💬 Your Sentence ({trans_info['language_label']})</div>
                    <div class="translation-text">{html.escape(analysis['raw_text'])}</div>
                </div>
                <div class="translation-card">
                    <div class="translation-label">🇮🇳 Devanagari Script</div>
                    <div class="translation-text">{html.escape(trans_info['native_script'])}</div>
                </div>
                <div class="translation-card">
                    <div class="translation-label">🇬🇧 English Meaning</div>
                    <div class="translation-text">{html.escape(trans_info['english'])}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif trans_info.get("error"):
            st.caption(f"ℹ️ Translation service note: {trans_info['error']}")

        # ---------------- 5. Morphology & Language Mix Columns ----------------
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### 🌱 Word Structure & Morphology")
            st.markdown("<p style='font-size:0.85rem; color:#64748B;'>Breaks down stems, affixes, plural endings, and tenses.</p>", unsafe_allow_html=True)
            
            morph_data = analysis["morphology"]["original"]
            # Convert to student friendly table format
            friendly_morph = []
            for m in morph_data:
                friendly_morph.append({
                    "Word": m["Token"],
                    "Root (Stem)": m["Root"],
                    "Suffix / Ending": m["Suffix"],
                    "Plural / Number": m["Number"],
                    "Tense / Feature": m["Other"]
                })
            st.dataframe(pd.DataFrame(friendly_morph), hide_index=True, use_container_width=True)

            if analysis["morphology"]["english"]:
                with st.expander("Show English translation morphology"):
                    st.dataframe(pd.DataFrame(analysis["morphology"]["english"]), hide_index=True, use_container_width=True)

        with c_right:
            st.markdown("#### 🔀 Language Mix (Code-Mixing)")
            lid_stats = analysis["language_id"]["stats"]
            hindi_cnt = lid_stats.get("Hindi Tokens", lid_stats.get("Marathi Tokens", 0))
            eng_cnt = lid_stats.get("English Tokens", 0)
            other_cnt = lid_stats.get("Other Tokens", 0)
            total_cnt = max(1, lid_stats.get("Total Tokens", 1))

            p_hi = int((hindi_cnt / total_cnt) * 100)
            p_en = int((eng_cnt / total_cnt) * 100)
            p_ot = max(0, 100 - p_hi - p_en)

            st.markdown(f"""
            <div class="mix-bar-container" style="padding:1rem 1.25rem; margin-top:0;">
                <div style="font-size:0.95rem; font-weight:700; color:#1E293B;">Language Distribution</div>
                <div class="mix-progress-track">
                    <div style="width:{p_hi}%; background:#38BDF8;" title="Hindi: {p_hi}%"></div>
                    <div style="width:{p_en}%; background:#A855F7;" title="English: {p_en}%"></div>
                    <div style="width:{p_ot}%; background:#94A3B8;" title="Punctuation/Other: {p_ot}%"></div>
                </div>
                <div class="mix-legend">
                    <div><span class="mix-dot" style="background:#38BDF8;"></span>Hindi: {p_hi}% ({hindi_cnt} words)</div>
                    <div><span class="mix-dot" style="background:#A855F7;"></span>English: {p_en}% ({eng_cnt} words)</div>
                    <div><span class="mix-dot" style="background:#94A3B8;"></span>Other: {p_ot}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            cmi_val = lid_stats.get("Code-Mixing Index (CMI)", "0.0%")
            switch_pts = lid_stats.get("Switch Points", 0)
            dom_lang = lid_stats.get("Dominant Language", "Unknown")

            if float(cmi_val.replace("%", "")) == 0.0:
                diagnosis = f"**Monolingual ({dom_lang})**: No code-mixing detected (CMI: {cmi_val})."
            else:
                diagnosis = f"**Code-Mixed**: CMI is **{cmi_val}** with **{switch_pts} switch points** (Dominant matrix: {dom_lang})."

            st.info(f"📊 **Code-Mixing Index (Gambäck & Das):** {diagnosis}")
            
            with st.expander("View token-by-token language table"):
                st.dataframe(pd.DataFrame(analysis["language_id"]["results"]), hide_index=True, use_container_width=True)

        # ---------------- 6. Named Entity Recognition ----------------
        st.markdown("#### 🔎 People, Places & Organizations (NER)")
        entities = analysis["entities"]["original"]
        if entities:
            ent_cols = st.columns(min(4, max(1, len(entities))))
            for i, ent in enumerate(entities):
                with ent_cols[i % len(ent_cols)]:
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:0.85rem; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.02); margin-bottom:0.5rem;">
                        <span style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">{ent['Entity Type']}</span>
                        <div style="font-size:1.15rem; font-weight:700; color:#0F172A; margin-top:0.2rem;">{ent['Entity']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No named people, places, or organizations were found in this sentence. Try adding names like 'Rahul' or places like 'Pune'!")

# ================================================================
# TAB 3: PRACTICE MODE
# ================================================================
elif current_tab == "🎮 Practice":
    st.markdown("### 🎮 Grammar Practice Playground")
    st.markdown("<p style='color:#64748B; font-size:0.95rem; margin-top:-0.5rem;'>Test your grammar skills on the sentences you speak every day!</p>", unsafe_allow_html=True)

    # Score Board
    stat_c1, stat_c2, stat_c3 = st.columns(3)
    with stat_c1:
        st.metric("🏆 Score", f"{st.session_state.quiz_score} pts")
    with stat_c2:
        accuracy = int((st.session_state.quiz_score / max(1, st.session_state.quiz_attempts)) * 100) if st.session_state.quiz_attempts > 0 else 100
        st.metric("🎯 Accuracy", f"{accuracy}%")
    with stat_c3:
        st.metric("📝 Questions Answered", f"{st.session_state.quiz_attempts}")

    # Generate questions from current analysis
    questions = generate_practice_questions(analysis)
    q_idx = st.session_state.current_q_idx % len(questions)
    current_q = questions[q_idx]

    st.markdown(f"""
    <div class="quiz-card">
        <span style="background:#E0F2FE; color:#0284C7; font-weight:700; font-size:0.8rem; padding:0.25rem 0.65rem; border-radius:16px;">
            {current_q.get('title', 'Question')}
        </span>
        <div style="font-size:1.3rem; font-weight:700; color:#0F172A; margin:0.85rem 0 1.25rem;">
            {current_q['question']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    ans_choice = st.radio(
        "Select your answer:",
        current_q["options"],
        key=f"quiz_opt_{current_q['id']}_{q_idx}",
        index=None
    )

    btn_c1, btn_c2 = st.columns([1, 3])
    with btn_c1:
        submit_btn = st.button("Check Answer ✨", type="primary", use_container_width=True)

    if submit_btn:
        if not ans_choice:
            st.warning("Please choose an option first!")
        else:
            st.session_state.quiz_attempts += 1
            st.session_state.quiz_submitted = True
            st.session_state.user_selected_ans = ans_choice

            if ans_choice == current_q["correct"]:
                st.session_state.quiz_score += 10
                st.success(f"{current_q['explanation_correct']}")
                st.balloons()
            else:
                st.error(f"{current_q['explanation_incorrect']}\n\n**Correct Answer:** {current_q['correct']}")

    if st.session_state.quiz_submitted:
        if st.button("Next Question ➡️", use_container_width=False):
            st.session_state.current_q_idx += 1
            st.session_state.quiz_submitted = False
            st.session_state.user_selected_ans = None
            st.rerun()

    st.divider()
    st.caption(f"Currently testing sentence: \"{st.session_state.sentence}\". Analyze another sentence in the Analyze tab to generate new customized questions!")

# ================================================================
# TAB 4: GRAMMAR GUIDE
# ================================================================
elif current_tab == "📖 Grammar Guide":
    st.markdown("### 📖 Visual Grammar Reference Guide")
    st.markdown("<p style='color:#64748B; font-size:0.95rem; margin-top:-0.5rem;'>Your friendly guide to Universal Dependencies parts of speech in English and Hinglish.</p>", unsafe_allow_html=True)

    # Category Filter
    selected_cat = st.radio(
        "Category Filter",
        list(GRAMMAR_CATEGORIES.keys()),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="guide_filter_radio",
    )

    allowed_tags = GRAMMAR_CATEGORIES.get(selected_cat)
    filtered_items = [
        item for item in GRAMMAR_GUIDE
        if allowed_tags is None or item["tag"] in allowed_tags
    ]

    for item in filtered_items:
        meta = TAG_METADATA.get(item["tag"], TAG_METADATA["X"])
        st.markdown(f"""
        <div class="guide-card" style="border-left: 6px solid {item['color']};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                <div style="font-size:1.25rem; font-weight:800; color:#0F172A; font-family:'Outfit', sans-serif;">
                    {item['emoji']} {item['name']}
                </div>
                <span style="background:{meta['bg']}; color:{meta['text']}; font-weight:700; font-size:0.78rem; padding:0.2rem 0.6rem; border-radius:12px;">
                    TAG: {item['tag']}
                </span>
            </div>
            <div style="font-size:0.95rem; color:#334155; line-height:1.5; margin-bottom:0.75rem;">
                {item['definition']}
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:0.85rem; font-size:0.88rem; margin-bottom:0.75rem;">
                <div style="color:#0284C7; font-weight:700; margin-bottom:0.25rem;">💡 Student Tip:</div>
                <div style="color:#475569;">{item['student_tip']}</div>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:0.75rem; font-size:0.86rem;">
                <div style="background:#EFF6FF; padding:0.65rem 0.85rem; border-radius:10px;">
                    <div style="font-weight:700; color:#1E40AF; margin-bottom:0.2rem;">Hinglish:</div>
                    <div>{item['code_mixed_example']}</div>
                    <div style="font-size:0.78rem; color:#6B7280; margin-top:0.2rem;">{item['code_mixed_notes']}</div>
                </div>
                <div style="background:#F0FDF4; padding:0.65rem 0.85rem; border-radius:10px;">
                    <div style="font-weight:700; color:#166534; margin-bottom:0.2rem;">Standard English:</div>
                    <div>{item['english_example']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# TAB 5: NLP LAB
# ================================================================
elif current_tab == "🧪 NLP Lab":
    st.markdown("### 🧪 Advanced NLP Research Lab")
    st.markdown("<p style='color:#64748B; font-size:0.95rem; margin-top:-0.5rem;'>Technical diagnostics, model arbitration controls, decision traces, and WordNet validation for researchers and educators.</p>", unsafe_allow_html=True)

    # 1. Model Health
    st.markdown("#### 1. Model Health & Status")
    health_c1, health_c2, health_c3 = st.columns(3)
    with health_c1:
        st.markdown(f"**Tier 1: Rules** {'✅ Calibrated' if inventory.rules_ok else '⚠️ Uncalibrated defaults'}")
        st.caption("Morphological rules & regional case markers")
    with health_c2:
        st.markdown(f"**Tier 3: HMM** {'✅ Loaded (hmm.npz)' if inventory.hmm_ok else '❌ Missing'}")
        st.caption("Second-order HMM with suffix smoothing")
    with health_c3:
        st.markdown(f"**Tier 2: Transformer (HingRoBERTa)** {'✅ Loaded' if inventory.transformer_ok else '⚠️ Disabled / Not loaded'}")
        st.caption("Fine-tuned Romanized Hinglish RoBERTa (l3cube-pune/hing-roberta)")

    st.divider()

    # 2. Live Arbiter Hyperparameters
    st.markdown("#### 2. Live Arbiter Configuration")
    st.caption("Tune the ensemble arbitration parameters in real-time. Changes immediately update predictions.")

    arb_c1, arb_c2, arb_c3 = st.columns(3)
    with arb_c1:
        new_thresh = st.slider(
            "Transformer Confidence Threshold",
            0.50, 0.99,
            float(st.session_state.threshold),
            0.01,
            help="Softmax threshold required for the transformer to win outright."
        )
        if new_thresh != st.session_state.threshold:
            st.session_state.threshold = new_thresh
            st.rerun()

    with arb_c2:
        new_fb = st.radio(
            "Fallback Policy",
            ["hmm", "max_conf"],
            index=0 if st.session_state.fallback == "hmm" else 1,
            format_func=lambda x: "Trust the HMM" if x == "hmm" else "Trust Max Confidence"
        )
        if new_fb != st.session_state.fallback:
            st.session_state.fallback = new_fb
            st.rerun()

    with arb_c3:
        new_over = st.checkbox(
            "Allow Confident Transformer to Overrule Rules",
            value=st.session_state.rule_overridable
        )
        if new_over != st.session_state.rule_overridable:
            st.session_state.rule_overridable = new_over
            st.rerun()

    st.divider()

    # 3. Decision Trace Table
    st.markdown("#### 3. Full Decision Trace & Attribution")
    if analysis and analysis.get("pos_results"):
        results = analysis["pos_results"]
        df_rows = []
        for r in results:
            df_rows.append({
                "Token": r["token"],
                "Final Tag": r["tag"],
                "Source Model": r["source"],
                "Confidence": r["conf"] if r["conf"] is not None else 0.0,
                "Rule": f"{r['candidates']['rule']['tag']} ({r['candidates']['rule']['conf']:.2f})" if r["candidates"]["rule"] else "—",
                "Transformer": f"{r['candidates']['transformer']['tag']} ({r['candidates']['transformer']['conf']:.2f})" if r["candidates"]["transformer"] else "—",
                "HMM": f"{r['candidates']['hmm']['tag']} ({r['candidates']['hmm']['conf']:.2f})" if r["candidates"]["hmm"] else "—",
                "Decision Rationale": r["reason"]
            })

        df_trace = pd.DataFrame(df_rows)
        st.dataframe(
            df_trace,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                )
            }
        )

        # Attribution shares
        shares = pd.Series([r["source"] for r in results]).value_counts()
        total_toks = len(results)
        share_summary = ", ".join([f"**{src}**: {cnt}/{total_toks} ({cnt/total_toks*100:.1f}%)" for src, cnt in shares.items()])
        st.info(f"📊 **Model Attribution Breakdown:** {share_summary}")

    # 4. WordNet POS Validation Layer
    st.markdown("#### 4. English POS Validation Layer (WordNet)")
    val_results = analysis.get("validation", [])
    if val_results:
        st.dataframe(pd.DataFrame(val_results), hide_index=True, use_container_width=True)
    else:
        st.caption("WordNet validation is computed on the translated English tokens.")

# ---------------------------------------------------------------- Footer
st.markdown("""
<div style="margin-top:3.5rem; padding:1.5rem 0; border-top:1px solid #E2E8F0; text-align:center; color:#94A3B8; font-size:0.85rem;">
    Grammar Buddy 🌱 · Code-Mixed NLP Research & Grammar Learning Platform · Universal Dependencies
</div>
""", unsafe_allow_html=True)
