# Evaluation on test set

160 sentences, 3476 tokens.

## Ablation

| System | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| HMM only | 0.8700 | 0.8392 | 0.8696 |
| Transformer only | 0.9194 | 0.8917 | 0.9193 |
| Rules → HMM | 0.8671 | 0.8339 | 0.8668 |
| Hybrid — blueprint (0.85, rules final) | 0.9148 | 0.8814 | 0.9143 |
| Hybrid — tuned on dev (0.94, hmm) | 0.9177 | 0.8917 | 0.9177 |

Rules alone fire on 28.6% of tokens with 95.6% precision.
HMM accuracy on OOV tokens: 51.8% (483 tokens).

## Which tier made the final call

| Source | Tokens | Share | Accuracy |
|---|---|---|---|
| transformer | 1813 | 52.2% | 94.2% |
| rule | 973 | 28.0% | 97.5% |
| transformer+hmm | 550 | 15.8% | 85.8% |
| hmm | 140 | 4.0% | 44.3% |

## Per-tag scores (final hybrid)

| Tag | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| X | 0.997 | 0.978 | 0.988 | 790 |
| VERB | 0.949 | 0.942 | 0.945 | 669 |
| NOUN | 0.863 | 0.926 | 0.893 | 516 |
| ADP | 0.926 | 0.936 | 0.931 | 346 |
| PROPN | 0.891 | 0.841 | 0.865 | 271 |
| PRON | 0.898 | 0.894 | 0.896 | 198 |
| PART | 0.860 | 0.883 | 0.871 | 188 |
| ADJ | 0.802 | 0.765 | 0.783 | 170 |
| DET | 0.872 | 0.879 | 0.876 | 116 |
| ADV | 0.818 | 0.810 | 0.814 | 100 |
| CCONJ | 0.957 | 0.857 | 0.904 | 77 |
| NUM | 0.875 | 1.000 | 0.933 | 35 |

## Most common errors (gold → predicted)

| Gold | Predicted | Count |
|---|---|---|
| VERB | NOUN | 21 |
| PROPN | NOUN | 19 |
| ADP | PART | 16 |
| ADJ | PROPN | 15 |
| ADJ | NOUN | 14 |
| NOUN | VERB | 12 |
| NOUN | ADJ | 11 |
| NOUN | PROPN | 10 |
| PART | NOUN | 8 |
| PRON | DET | 8 |
