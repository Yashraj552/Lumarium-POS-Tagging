# Transformer POS Tagger Benchmark: IndicBERT v2 vs HingRoBERTa

## 1. Executive Summary

Evaluation performed on the frozen test set (`data/test.json`) containing **160 sentences** and **3476 tokens**.

| Metric | IndicBERT v2 (Fine-Tuned) | HingRoBERTa (Fine-Tuned) | Difference |
|---|---|---|---|
| **Overall Accuracy** | 0.8838 | 0.9194 | +0.0356 (+3.56%) |
| **Macro-F1** | 0.8532 | 0.8917 | +0.0385 |
| **Weighted-F1** | 0.8838 | 0.9193 | +0.0355 |
| **Ambiguous Sentences Acc** | 0.8828 | 0.9174 | +0.0346 |
| **OOV Token Accuracy** | 0.7635 | 0.8556 | +0.0921 |
| **Inference Speed (tokens/s)** | 351.5 | 346.8 | -4.7 |
| **Latency (ms/sentence)** | 61.8 ms | 62.6 ms | - |

## 2. Per-Tag Performance Comparison (Precision / Recall / F1)

| Tag | Support | IndicBERT Precision | IndicBERT Recall | IndicBERT F1 | HingRoBERTa Precision | HingRoBERTa Recall | HingRoBERTa F1 | F1 Delta |
|---|---|---|---|---|---|---|---|---|
| `ADJ` | 170 | 0.705 | 0.729 | 0.717 | 0.808 | 0.741 | 0.773 | +0.056 |
| `ADP` | 346 | 0.942 | 0.931 | 0.936 | 0.936 | 0.965 | 0.950 | +0.014 |
| `ADV` | 100 | 0.705 | 0.790 | 0.745 | 0.847 | 0.830 | 0.838 | +0.093 |
| `AUX` | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| `CCONJ` | 77 | 0.861 | 0.883 | 0.872 | 0.929 | 0.844 | 0.884 | +0.013 |
| `DET` | 116 | 0.856 | 0.922 | 0.888 | 0.843 | 0.922 | 0.881 | -0.007 |
| `INTJ` | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| `NOUN` | 516 | 0.793 | 0.893 | 0.840 | 0.844 | 0.932 | 0.886 | +0.045 |
| `NUM` | 35 | 0.850 | 0.971 | 0.907 | 0.868 | 0.943 | 0.904 | -0.003 |
| `PART` | 188 | 0.785 | 0.835 | 0.809 | 0.894 | 0.856 | 0.875 | +0.066 |
| `PRON` | 198 | 0.831 | 0.894 | 0.861 | 0.923 | 0.904 | 0.913 | +0.052 |
| `PROPN` | 271 | 0.923 | 0.661 | 0.770 | 0.893 | 0.830 | 0.860 | +0.091 |
| `PUNCT` | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| `SCONJ` | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| `SYM` | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| `VERB` | 669 | 0.937 | 0.894 | 0.915 | 0.958 | 0.943 | 0.950 | +0.035 |
| `X` | 790 | 0.987 | 0.970 | 0.978 | 0.995 | 0.976 | 0.985 | +0.007 |

## 3. Ambiguous Homograph Disambiguation

| Token | Count | IndicBERT Acc | HingRoBERTa Acc | Delta |
|---|---|---|---|---|
| `aur` | 5 | 80.0% | 80.0% | +0.0% |
| `bhi` | 7 | 100.0% | 100.0% | +0.0% |
| `he` | 8 | 75.0% | 100.0% | +25.0% |
| `is` | 19 | 100.0% | 100.0% | +0.0% |
| `ki` | 32 | 78.1% | 75.0% | -3.1% |
| `koi` | 7 | 28.6% | 28.6% | +0.0% |
| `me` | 12 | 100.0% | 100.0% | +0.0% |
| `ne` | 6 | 100.0% | 100.0% | +0.0% |
| `par` | 5 | 60.0% | 80.0% | +20.0% |
| `se` | 16 | 100.0% | 100.0% | +0.0% |
| `to` | 92 | 87.0% | 84.8% | -2.2% |
| `us` | 7 | 57.1% | 85.7% | +28.6% |
| `wo` | 3 | 66.7% | 66.7% | +0.0% |
| `ye` | 9 | 88.9% | 88.9% | +0.0% |

## 4. Top Confusions (Gold -> Predicted)

### IndicBERT v2 Top Confusions
| Gold | Predicted | Count |
|---|---|---|
| `PROPN` | `NOUN` | 41 |
| `VERB` | `NOUN` | 32 |
| `ADJ` | `NOUN` | 23 |
| `PROPN` | `ADJ` | 17 |
| `NOUN` | `VERB` | 16 |
| `NOUN` | `ADJ` | 14 |
| `VERB` | `PART` | 13 |
| `ADP` | `PART` | 10 |
| `PROPN` | `PRON` | 10 |
| `PART` | `NOUN` | 10 |

### HingRoBERTa Top Confusions
| Gold | Predicted | Count |
|---|---|---|
| `PROPN` | `NOUN` | 25 |
| `VERB` | `NOUN` | 23 |
| `ADJ` | `NOUN` | 20 |
| `ADJ` | `PROPN` | 15 |
| `NOUN` | `ADJ` | 12 |
| `NOUN` | `PROPN` | 10 |
| `PRON` | `DET` | 10 |
| `PART` | `NOUN` | 9 |
| `NOUN` | `VERB` | 9 |
| `PART` | `VERB` | 7 |
