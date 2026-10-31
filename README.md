# 🌟 Lumarium: Hybrid POS Tagger & Linguistic Observatory

A hybrid Natural Language Processing (NLP) system for Part-of-Speech (POS) tagging of Hindi-English (Hinglish) code-mixed text, combining calibrated linguistic rules, pre-trained & fine-tuned Transformers, and Hidden Markov Models with dynamic confidence arbitration.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Uvicorn-009688.svg)](https://fastapi.tiangolo.com/)
[![Primary Model](https://img.shields.io/badge/Primary%20Transformer-HingRoBERTa%20(91.94%25%20Acc)-purple.svg)](https://huggingface.co/l3cube-pune/hing-roberta)
[![Baseline Model](https://img.shields.io/badge/Baseline-IndicBERT%20v2-orange.svg)](https://huggingface.co/ai4bharat/IndicBERTv2-alpha-POS-tagging)
[![Dataset](https://img.shields.io/badge/Benchmark-LinCE%20Hindi--English%20POS-success.svg)](https://huggingface.co/datasets/lince-benchmark/lince)
[![Tagset](https://img.shields.io/badge/Tagset-Universal%20Dependencies%20(17%20Tags)-informational.svg)](https://universaldependencies.org/u/pos/)
[![Tests](https://img.shields.io/badge/Tests-19%20Passing%20(Pytest)-brightgreen.svg)]()

> **Illuminate every word.** Unifying deterministic morphological rules, probabilistic sequence modelling, and state-of-the-art code-mixed Transformer architectures for explainable code-mixed language understanding.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Benchmark & Empirical Evaluation Results](#-benchmark--empirical-evaluation-results)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation and Quick Start](#-installation-and-quick-start)
- [Running the Applications](#-running-the-applications)
- [Model Fine-Tuning & Evaluation](#-model-fine-tuning--evaluation)
- [GitHub Push Preparation](#-github-push-preparation)
- [Limitations & Future Roadmap](#-limitations--future-roadmap)
- [References & Citations](#-references--citations)

---

## 📖 Overview

Code-mixed language (such as **Hinglish**, the fluid alternating use of Hindi and English) is the dominant communication medium across Indian digital platforms, social media, and messaging apps. However, POS tagging for Romanized code-mixed text presents significant NLP challenges:

1. **Orthographic Variation:** Non-standard, informal Romanized spellings (e.g., *khelunga*, *khelunga*, *khelooo*).
2. **Cross-Lingual Homographs:** Identical character sequences that serve completely different syntactic roles depending on language and context:
   - *he* (English 3rd-person pronoun `PRON` vs. Hindi emphatic particle / auxiliary `PART`/`AUX`).
   - *to* (English preposition `ADP` vs. Hindi discourse particle `PART`).
   - *par* (Hindi adposition `ADP` "on" vs. coordinating conjunction `CCONJ` "but").
   - *us* (English objective pronoun `PRON` vs. Hindi oblique demonstrative stem `PRON`).
3. **Morphological Hybridization:** English lexical roots inflected with Hindi grammatical morphemes (e.g., *dog-ne*, *video-ka*, *run-kar*).
4. **Out-of-Vocabulary (OOV) Sparsity:** Frequent unseen slang, colloquialisms, and regional affixes.

**Lumarium** addresses these challenges via a multi-tier hybrid ensemble that combines:
- **Tier 1 (Rule-Based Tagger):** Calibrated deterministic regex patterns, token-shape classifiers, and morphological affix rules.
- **Tier 2 (Transformer Tagger):** **HingRoBERTa** (`l3cube-pune/hing-roberta`), pretrained on 4.1M Romanized code-mixed sentences and fine-tuned for token classification with fast subword alignment.
- **Tier 3 (HMM Tagger):** Second-order Hidden Markov Model with TnT-style suffix probability smoothing and Viterbi sequence decoding.
- **Tier 4 (Multi-Tier Arbiter):** A token-level waterfall decision arbiter that cross-examines confidence scores, calibrated precision, OOV indicators, and configurable fallback policies.

---

## 🏗️ System Architecture

```text
                        ┌──────────────────────────────────────┐
                        │   Input Code-Mixed Sentence (Text)   │
                        └──────────────────┬───────────────────┘
                                           │
                                           ▼
                        ┌──────────────────────────────────────┐
                        │     Tokenization & Normalization     │
                        │ (Punctuation, Emojis, Elongations)   │
                        └──────────────────┬───────────────────┘
                                           │
           ┌───────────────────────────────┼───────────────────────────────┐
           │                               │                               │
           ▼                               ▼                               ▼
 ┌───────────────────┐           ┌───────────────────┐           ┌───────────────────┐
 │   Tier 1: Rules   │           │Tier 2: Transformer│           │    Tier 3: HMM    │
 │ Morphological,    │           │   HingRoBERTa     │           │ Second-Order HMM, │
 │ Shape & Lexicon   │           │ (Code-Mixed BPE)  │           │  Suffix Smoothing │
 └─────────┬─────────┘           └─────────┬─────────┘           └─────────┬─────────┘
           │ (Tag, Calibrated              │ (Tag, Softmax                 │ (Tag, Viterbi
           │  Precision)                   │  Confidence)                  │  Posterior)
           └───────────────────────────────┼───────────────────────────────┘
                                           │
                                           ▼
                        ┌──────────────────────────────────────┐
                        │       Multi-Tier Hybrid Arbiter      │
                        │   - Rule Precision vs Softmax Conf   │
                        │   - Threshold Check (th = 0.94)      │
                        │   - HMM OOV Detection & Fallback     │
                        └──────────────────┬───────────────────┘
                                           │
                                           ▼
                        ┌──────────────────────────────────────┐
                        │       Final POS Tag Sequence         │
                        │  + Full Decision Trace & Attribution │
                        └──────────────────┬───────────────────┘
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
┌──────────────────────────┐                               ┌──────────────────────────┐
│   FastAPI Web Service    │                               │    Streamlit Research    │
│ & Modern HTML5 Frontend  │                               │        Dashboard         │
│  (http://localhost:8001) │                               │  (http://localhost:8501) │
└──────────────────────────┘                               └──────────────────────────┘
```

---

## 📊 Benchmark & Empirical Evaluation Results

All quantitative benchmarks are evaluated on the exact same frozen test set from the **LinCE Hindi-English POS Benchmark** (160 sentences, 3,476 tokens), mapped into the **17 Universal Dependencies (UD) POS tags**.

### Overall System Ablation

| System / Model Tier | Accuracy | Macro-F1 | Weighted-F1 | Ambiguous Sents Acc | OOV Token Acc |
|---|:---:|:---:|:---:|:---:|:---:|
| **HMM Only** | 87.00% | 0.8392 | 0.8696 | 86.72% | 51.80% |
| **Rules → HMM** | 86.71% | 0.8339 | 0.8668 | 86.50% | 51.80% |
| **IndicBERT v2 (Fine-Tuned Baseline)** | 88.38% | 0.8532 | 0.8838 | 88.28% | 76.35% |
| **HingRoBERTa (Primary Model)** | **91.94%** | **0.8917** | **0.9193** | **91.74%** | **85.56%** |
| **Hybrid, Blueprint Configuration (th=0.85)** | 91.48% | 0.8814 | 0.9143 | 91.26% | 84.10% |
| **Tuned Hybrid (Arbiter th=0.94, HMM fallback)** | **91.77%** | **0.8917** | **0.9177** | **91.58%** | **85.20%** |

### Head-to-Head Comparison: IndicBERT v2 vs HingRoBERTa

A comprehensive evaluation ([`results/indicbert_vs_hingroberta_report.md`](results/indicbert_vs_hingroberta_report.md)) demonstrates the empirical advantages of using a Romanized code-mix-pretrained model:

- **Overall Accuracy:** **+3.56% absolute improvement** (91.94% vs 88.38%).
- **Macro-F1:** **+0.0385 improvement** (0.8917 vs 0.8532).
- **Out-of-Vocabulary (OOV) Robustness:** **+9.21% gain** on words unseen in training (85.56% vs 76.35%).
- **Ambiguous Sentence Accuracy:** **+3.46% gain** across complex cross-lingual sentences (91.74% vs 88.28%).
- **CPU Inference Speed:** Virtually identical throughput (~62 ms/sentence, ~350 tokens/second).

### Per-Tag Performance Highlights

| Tag | Support | IndicBERT F1 | HingRoBERTa F1 | F1 Delta | Primary Factor |
|---|:---:|:---:|:---:|:---:|---|
| `PROPN` | 271 | 0.770 | **0.860** | **+0.091** | Reduced `PROPN` &rarr; `NOUN` misclassifications from 41 down to 25. |
| `ADV` | 100 | 0.745 | **0.838** | **+0.093** | Correctly disambiguates Hinglish adverbs (*fast*, *so*, *bohot*). |
| `PART` | 188 | 0.809 | **0.875** | **+0.066** | Accurately identifies emphatic discourse markers (*to*, *bhi*, *yaar*). |
| `ADJ` | 170 | 0.717 | **0.773** | **+0.056** | Sharper code-switching boundaries for code-mixed English adjectives. |
| `PRON` | 198 | 0.861 | **0.913** | **+0.052** | Disambiguates inflected Hindi pronouns (*usne*, *usko*, *main*). |
| `VERB` | 669 | 0.915 | **0.950** | **+0.035** | Better recognition of inflected stems (*bola*, *dekha*, *aaunga*). |

---

## ✨ Key Features

- **Primary Transformer Head:** Fine-tuned `l3cube-pune/hing-roberta` with dynamic batch padding and subword-to-word probability pooling.
- **Preserved Dual Model Support:** Both `models/hingroberta_pos` and `models/indicbert_pos` can be loaded or compared on demand.
- **The NLP Observatory:** Interactive inspection dashboard featuring real-time token-level decision traces, model attribution breakdowns, and WordNet lexical validation.
- **Live Arbiter Controls:** Real-time threshold tuning (`0.50` to `0.99`), fallback policy selection (`Trust HMM` vs `Trust Max Confidence`), and rule-overriding toggles without server restart.
- **Transliteration & Translation:** Automated Hindi-English code-mixed translation with native Devanagari transliteration display.
- **Interactive Practice & UD Compendium:** Built-in grammar quiz generator and a 17-tag Universal Dependencies reference guide.
- **100% Offline Inference:** Model loading and prediction caching are fully self-contained without mandatory cloud connections.

---

## 🛠️ Technology Stack

- **Core & Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic
- **Deep Learning & NLP:** PyTorch 2.1+, Hugging Face Transformers, NLTK, spaCy
- **Sequence Modelling:** Hidden Markov Models (HMM), Viterbi Decoding, Add-$k$ Smoothing
- **Dashboard UI:** Streamlit
- **Web Frontend:** HTML5, CSS3 (Modern Scholarly Theme), Vanilla ES6+ JavaScript
- **Benchmarking & Testing:** Pytest, LinCE Hindi-English POS Corpus

---

## 📁 Project Structure

```text
Lumarium-Hybrid-POS-Tagger/
├── backend/
│   ├── main.py                  # FastAPI server entry point
│   ├── api/                     # REST endpoints (analyze, health, guide, practice)
│   └── services/
│       └── nlp_service.py       # Singleton pipeline orchestration layer
│
├── frontend/
│   ├── index.html               # Lumarium Web Interface & NLP Observatory
│   ├── css/                     # Scholarly design system (main, components, responsive)
│   └── js/                      # Frontend controllers (app, analyze, components, practice)
│
├── components/
│   ├── grammar_guide_data.py    # Universal Dependencies reference compendium
│   ├── nlp_bridge.py            # Streamlit-to-pipeline bridge
│   ├── quiz_generator.py        # Dynamic grammar practice question generator
│   └── ui_styles.py             # Streamlit design tokens
│
├── data/
│   ├── train.json               # LinCE Hinglish training split (927 sentences)
│   ├── dev.json                 # LinCE Hinglish validation split (103 sentences)
│   └── test.json                # LinCE Hinglish frozen evaluation split (160 sentences)
│
├── models/
│   ├── hingroberta_pos/         # Primary fine-tuned HingRoBERTa model checkpoint
│   ├── indicbert_pos/           # Fine-tuned IndicBERT v2 model checkpoint
│   ├── arbiter.json             # Dev-tuned arbitration hyperparameters
│   ├── hmm.npz                  # Trained HMM transition & emission matrices
│   ├── rules.json               # Calibrated rule precision configurations
│   ├── transformer_dev.json     # Cached predictions on dev split
│   └── transformer_test.json    # Cached predictions on test split
│
├── results/
│   ├── indicbert_vs_hingroberta_report.md  # Detailed head-to-head comparison
│   ├── model_comparison.json               # Quantitative benchmark metrics
│   ├── metrics.json                        # Ablation metrics
│   └── report.md                           # System ablation report
│
├── tests/
│   ├── test_pipeline.py         # 15 core pipeline and unit tests
│   └── test_new_features.py     # 4 integration and feature tests
│
├── app.py                       # Streamlit interactive research dashboard
├── compare_models.py            # Head-to-head model comparison benchmark script
├── data_prep.py                 # LinCE CSV to JSON dataset converter
├── ensemble.py                  # Arbiter logic and grid-search tuning
├── evaluate.py                  # Test set evaluation and ablation report generator
├── finetune_hingroberta.py      # HingRoBERTa fine-tuning pipeline
├── finetune_indicbert.py        # IndicBERT v2 fine-tuning pipeline
├── hmm_tagger.py                # Second-order HMM tagger with suffix smoothing
├── language_identifier.py       # Code-switching language identification
├── morphology.py                # Morphological analysis heuristics
├── ner.py                       # Named entity extraction
├── pipeline.py                  # Model loaders and caching utilities
├── rule_tagger.py               # Deterministic rule engine and calibrator
├── run_all.py                   # Automated end-to-end pipeline executor
├── tagset.py                    # Universal Dependencies 17-tag normalizer
├── text_utils.py                # Code-mixed tokenization and cleaning
├── train.py                     # Rules calibration + HMM fit + Arbiter tuner
└── translate.py                 # Transliteration and English translation
```

---

## 🚀 Installation and Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/Lumarium-Hybrid-POS-Tagger.git
cd Lumarium-Hybrid-POS-Tagger
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 💻 Running the Applications

### Option A: Modern Web Application + NLP Observatory (Recommended)
Launch the FastAPI backend and HTML5 frontend:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```
Open your browser and navigate to:
**[http://127.0.0.1:8001/](http://127.0.0.1:8001/)**

### Option B: Streamlit Research Dashboard
Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py
```
Open your browser at:
**[http://localhost:8501/](http://localhost:8501/)**

### Option C: Run the Full Test Suite
Verify that all unit tests, token alignments, and arbitration mechanics pass:
```bash
pytest -v
```

---

## 🧪 Model Fine-Tuning & Evaluation

### 1. Fine-Tune HingRoBERTa (Primary Model)
```bash
python finetune_hingroberta.py --epochs 4 --batch-size 16 --lr 3e-5
```
This trains `l3cube-pune/hing-roberta` with dynamic sequence batch padding, validates against `data/dev.json` using Macro-F1 checkpoint selection, and saves the best model to `models/hingroberta_pos`.

### 2. Fine-Tune IndicBERT v2 (Baseline Model)
```bash
python finetune_indicbert.py
```
Saves the fine-tuned IndicBERT v2 checkpoint to `models/indicbert_pos`.

### 3. Head-to-Head Comparative Benchmark
To run an empirical comparison between IndicBERT and HingRoBERTa on the frozen test set:
```bash
python compare_models.py
```
Generates [`results/indicbert_vs_hingroberta_report.md`](results/indicbert_vs_hingroberta_report.md) and [`results/model_comparison.json`](results/model_comparison.json).

### 4. Calibrate Rules & Retune Hybrid Arbiter
```bash
python train.py
```
Calibrates rule precision on `data/train.json`, fits the HMM transition/emission matrices (`models/hmm.npz`), and executes grid-search arbiter tuning on `data/dev.json` (`models/arbiter.json`).

### 5. Final Evaluation on Frozen Test Set
```bash
python evaluate.py
```
Outputs the ablation table and detailed tier breakdown to [`results/report.md`](results/report.md).

---

## 📦 GitHub Push Preparation

Before pushing this repository to GitHub, verify that your `.gitignore` excludes heavy binary weights:

1. **Large Weights Protection:**
   The repository `.gitignore` automatically excludes `models/hingroberta_pos/*.safetensors` (~1.1 GB) and `models/indicbert_pos/` to comply with GitHub's 100MB file limit. The configuration files, tokenizers, vocabularies, and test prediction caches are tracked so anyone can clone and immediately run inference or retrain using the provided scripts.

2. **Git Commands:**
   ```bash
   git init
   git add .
   git commit -m "feat: complete Lumarium hybrid Hinglish POS tagger with HingRoBERTa"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/Lumarium-Hybrid-POS-Tagger.git
   git push -u origin main
   ```

---

## ⚠️ Limitations & Future Roadmap

- **Emphatic Particles:** The token *to* remains partially ambiguous when used as an emphatic discourse marker (*tum to pagal ho*) vs. an adposition (*welcome to mumbai*).
- **Phonetic Spelling Variants:** Extremely varied phonetic spellings (*sooooo*, *plzzzz*) rely on rule-based character compression prior to subword tokenization.
- **Sequence-Level CRF Head:** Investigating whether a Linear-Chain CRF decoding layer over HingRoBERTa embeddings can further resolve rare transition errors.

---

## 📖 References & Citations

1. **LinCE Benchmark:** Aguilar et al., *LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation*, LREC 2020.
2. **HingRoBERTa Pretraining:** Nayak & Joshi, *L3Cube-HingCorpus and HingRoBERTa: Pretrained Language Models for Hinglish Code-Mixed Text*, 2022.
3. **IndicBERT:** Doddapaneni et al., *IndicBERT: A Multilingual Pre-trained Model for Indian Languages*, 2021.
4. **Universal Dependencies:** Nivre et al., *Universal Dependencies v2: An Infrastructure for Cross-Lingual Evaluation*, LREC 2020.

---

<div align="center">
  <sub>Developed for academic research and code-mixed natural language understanding.</sub>
</div>
