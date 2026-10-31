"""
compare_models.py — Rigorous Head-to-Head Comparison: IndicBERT v2 vs HingRoBERTa.

Evaluates both fine-tuned models on the exact same frozen test set (data/test.json).
Computes:
- Overall Accuracy, Macro-F1, Weighted-F1
- Per-tag Precision, Recall, F1 for all 17 UD tags
- Confusion matrix (top confusion pairs)
- Ambiguous sentence performance (homographs and code-switching contexts)
- OOV token performance (tokens unseen in data/train.json)
- Inference throughput and latency (ms/sentence, tokens/sec)
- Qualitative error breakdown

Outputs:
- results/indicbert_vs_hingroberta_report.md
- results/model_comparison.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from metrics import confusions, score
from tagset import UD_TAGS, normalize_tag
from text_utils import normalize_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AMBIGUOUS_HOMOGRAPHS = frozenset({
    "ki", "to", "par", "koi", "he", "is", "us", "me", "aur", "ye", "wo", "se", "ne", "bhi"
})


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def align_subwords(word_ids: Sequence[int | None], probs: np.ndarray,
                   n_words: int, mode: str = "first") -> list[tuple[int, float] | None]:
    if mode == "first":
        out: list[tuple[int, float] | None] = [None] * n_words
        for pos, wid in enumerate(word_ids):
            if wid is None or wid >= n_words or out[wid] is not None:
                continue
            label = int(np.argmax(probs[pos]))
            out[wid] = (label, float(probs[pos, label]))
        return out
    else:  # mean pooling
        word_probs: list[list[np.ndarray]] = [[] for _ in range(n_words)]
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid < n_words:
                word_probs[wid].append(probs[pos])
        out = [None] * n_words
        for wid, plist in enumerate(word_probs):
            if not plist:
                continue
            mean_p = np.mean(plist, axis=0)
            label = int(np.argmax(mean_p))
            out[wid] = (label, float(mean_p[label]))
        return out


class EvaluatorModel:
    def __init__(self, model_path: str | Path, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
        path = str(model_path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
            self.tokenizer(["probe"], is_split_into_words=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, add_prefix_space=True, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True).to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.id2tag = {i: normalize_tag(lbl) for i, lbl in self.id2label.items()}

    def predict_sentences(self, sentences: list[list[str]], pooling_mode: str = "first") -> list[list[dict]]:
        results = []
        for words in sentences:
            if not words:
                results.append([])
                continue
            clean_tokens = [
                normalize_token(t) if not (t.startswith("@") or t.startswith("#") or t.startswith("http")) else t
                for t in words
            ]
            enc = self.tokenizer(
                clean_tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            word_ids = enc.word_ids(0)
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
            probs = softmax(logits.float().cpu().numpy())
            aligned = align_subwords(word_ids, probs, len(words), mode=pooling_mode)

            sent_res = []
            for tok, hit in zip(words, aligned):
                if hit is None:
                    sent_res.append({"token": tok, "tag": "X", "conf": 0.0, "source": "transformer"})
                else:
                    lid, p = hit
                    sent_res.append({"token": tok, "tag": self.id2tag[lid], "conf": round(p, 4), "source": "transformer"})
            results.append(sent_res)
        return results


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Running Benchmark on device: {device}")

    # Load frozen test set
    test_data = json.loads(Path("data/test.json").read_text(encoding="utf-8"))
    train_data = json.loads(Path("data/train.json").read_text(encoding="utf-8"))

    # Vocabulary for OOV analysis
    train_vocab = {normalize_token(w["token"]).lower() for s in train_data for w in s}

    test_tokens = [[w["token"] for w in s] for s in test_data]
    gold_flat = [normalize_tag(w["tag"]) for s in test_data for w in s]
    tokens_flat = [w["token"] for s in test_data for w in s]

    # Identify ambiguous sentences and OOV indices
    ambiguous_sent_indices = set()
    for i, s in enumerate(test_data):
        tokens_in_s = {normalize_token(w["token"]).lower() for w in s}
        if tokens_in_s & AMBIGUOUS_HOMOGRAPHS:
            ambiguous_sent_indices.add(i)

    oov_indices = [
        i for i, tok in enumerate(tokens_flat)
        if normalize_token(tok).lower() not in train_vocab
    ]

    print(f"--> Test Set: {len(test_data)} sentences, {len(gold_flat)} tokens")
    print(f"--> Ambiguous Sentences: {len(ambiguous_sent_indices)} / {len(test_data)}")
    print(f"--> OOV Tokens: {len(oov_indices)} / {len(gold_flat)} ({len(oov_indices)/len(gold_flat):.1%})")

    models_to_test = [
        ("IndicBERT v2 (Fine-Tuned)", Path("models/indicbert_pos")),
        ("HingRoBERTa (Fine-Tuned)", Path("models/hingroberta_pos")),
    ]

    benchmark_results = {}

    for name, path in models_to_test:
        if not path.exists():
            print(f"[!] Path {path} does not exist, skipping {name}...")
            continue

        print(f"\n========================================================")
        print(f"Evaluating: {name} from {path}")
        evaluator = EvaluatorModel(path, name=name, device=device)

        # Warmup
        _ = evaluator.predict_sentences(test_tokens[:5])

        # Timed Inference
        t0 = time.time()
        preds = evaluator.predict_sentences(test_tokens, pooling_mode="first")
        inf_duration = time.time() - t0

        ms_per_sent = (inf_duration / len(test_tokens)) * 1000
        tokens_per_sec = len(gold_flat) / inf_duration

        preds_flat = [p["tag"] for s in preds for p in s]
        conf_flat = [p["conf"] for s in preds for p in s]

        # Overall scores
        overall_scores = score(gold_flat, preds_flat)

        # Ambiguous sentence accuracy
        ambig_gold, ambig_preds = [], []
        for idx, (sent_gold, sent_pred) in enumerate(zip(test_data, preds)):
            if idx in ambiguous_sent_indices:
                ambig_gold.extend([normalize_tag(w["tag"]) for w in sent_gold])
                ambig_preds.extend([w["tag"] for w in sent_pred])
        ambig_scores = score(ambig_gold, ambig_preds)

        # OOV accuracy
        oov_correct = sum(preds_flat[idx] == gold_flat[idx] for idx in oov_indices)
        oov_acc = oov_correct / len(oov_indices) if oov_indices else 0.0

        # Top confusions
        top_confusions = confusions(gold_flat, preds_flat, top=10)

        # Specific homograph accuracy breakdown
        homograph_stats = {}
        for hg in sorted(AMBIGUOUS_HOMOGRAPHS):
            hg_indices = [
                i for i, tok in enumerate(tokens_flat)
                if normalize_token(tok).lower() == hg
            ]
            if hg_indices:
                hg_correct = sum(preds_flat[i] == gold_flat[i] for i in hg_indices)
                homograph_stats[hg] = {
                    "count": len(hg_indices),
                    "correct": hg_correct,
                    "accuracy": round(hg_correct / len(hg_indices), 4),
                    "gold_distribution": dict(Counter(gold_flat[i] for i in hg_indices)),
                    "pred_distribution": dict(Counter(preds_flat[i] for i in hg_indices)),
                }

        benchmark_results[name] = {
            "overall": {
                "accuracy": round(overall_scores["accuracy"], 4),
                "macro_f1": round(overall_scores["macro_f1"], 4),
                "weighted_f1": round(overall_scores["weighted_f1"], 4),
            },
            "per_tag": overall_scores["per_tag"],
            "ambiguous_sentences": {
                "count": len(ambiguous_sent_indices),
                "accuracy": round(ambig_scores["accuracy"], 4),
                "macro_f1": round(ambig_scores["macro_f1"], 4),
            },
            "oov": {
                "count": len(oov_indices),
                "accuracy": round(oov_acc, 4),
            },
            "speed": {
                "total_inference_time_seconds": round(inf_duration, 3),
                "ms_per_sentence": round(ms_per_sent, 2),
                "tokens_per_second": round(tokens_per_sec, 1),
            },
            "top_confusions": top_confusions,
            "homographs": homograph_stats,
            "predictions": preds,
        }

        print(f"  Accuracy:            {overall_scores['accuracy']:.4f}")
        print(f"  Macro-F1:            {overall_scores['macro_f1']:.4f}")
        print(f"  Weighted-F1:         {overall_scores['weighted_f1']:.4f}")
        print(f"  Ambiguous Sents Acc: {ambig_scores['accuracy']:.4f}")
        print(f"  OOV Tokens Acc:      {oov_acc:.4f}")
        print(f"  Speed:               {ms_per_sent:.1f} ms/sent | {tokens_per_sec:.1f} tok/s")

    # Save JSON results
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Strip predictions from JSON summary for compact persistence
    json_summary = {}
    for k, v in benchmark_results.items():
        json_summary[k] = {ik: iv for ik, iv in v.items() if ik != "predictions"}
    (out_dir / "model_comparison.json").write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    # Generate Markdown Report
    lines = [
        "# Transformer POS Tagger Benchmark: IndicBERT v2 vs HingRoBERTa",
        "",
        "## 1. Executive Summary",
        "",
        f"Evaluation performed on the frozen test set (`data/test.json`) containing **{len(test_data)} sentences** and **{len(gold_flat)} tokens**.",
        "",
        "| Metric | IndicBERT v2 (Fine-Tuned) | HingRoBERTa (Fine-Tuned) | Difference |",
        "|---|---|---|---|",
    ]

    ib_res = benchmark_results.get("IndicBERT v2 (Fine-Tuned)")
    hr_res = benchmark_results.get("HingRoBERTa (Fine-Tuned)")

    if ib_res and hr_res:
        diff_acc = hr_res["overall"]["accuracy"] - ib_res["overall"]["accuracy"]
        diff_mf1 = hr_res["overall"]["macro_f1"] - ib_res["overall"]["macro_f1"]
        diff_wf1 = hr_res["overall"]["weighted_f1"] - ib_res["overall"]["weighted_f1"]
        diff_amb = hr_res["ambiguous_sentences"]["accuracy"] - ib_res["ambiguous_sentences"]["accuracy"]
        diff_oov = hr_res["oov"]["accuracy"] - ib_res["oov"]["accuracy"]
        diff_spd = hr_res["speed"]["tokens_per_second"] - ib_res["speed"]["tokens_per_second"]

        lines.extend([
            f"| **Overall Accuracy** | {ib_res['overall']['accuracy']:.4f} | {hr_res['overall']['accuracy']:.4f} | {diff_acc:+.4f} ({'+' if diff_acc>=0 else ''}{diff_acc*100:.2f}%) |",
            f"| **Macro-F1** | {ib_res['overall']['macro_f1']:.4f} | {hr_res['overall']['macro_f1']:.4f} | {diff_mf1:+.4f} |",
            f"| **Weighted-F1** | {ib_res['overall']['weighted_f1']:.4f} | {hr_res['overall']['weighted_f1']:.4f} | {diff_wf1:+.4f} |",
            f"| **Ambiguous Sentences Acc** | {ib_res['ambiguous_sentences']['accuracy']:.4f} | {hr_res['ambiguous_sentences']['accuracy']:.4f} | {diff_amb:+.4f} |",
            f"| **OOV Token Accuracy** | {ib_res['oov']['accuracy']:.4f} | {hr_res['oov']['accuracy']:.4f} | {diff_oov:+.4f} |",
            f"| **Inference Speed (tokens/s)** | {ib_res['speed']['tokens_per_second']:.1f} | {hr_res['speed']['tokens_per_second']:.1f} | {diff_spd:+.1f} |",
            f"| **Latency (ms/sentence)** | {ib_res['speed']['ms_per_sentence']:.1f} ms | {hr_res['speed']['ms_per_sentence']:.1f} ms | - |",
            "",
            "## 2. Per-Tag Performance Comparison (Precision / Recall / F1)",
            "",
            "| Tag | Support | IndicBERT Precision | IndicBERT Recall | IndicBERT F1 | HingRoBERTa Precision | HingRoBERTa Recall | HingRoBERTa F1 | F1 Delta |",
            "|---|---|---|---|---|---|---|---|---|",
        ])

        all_tags = sorted(UD_TAGS)
        for t in all_tags:
            ib_pt = ib_res["per_tag"].get(t, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
            hr_pt = hr_res["per_tag"].get(t, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
            sup = ib_pt["support"] or hr_pt["support"]
            f1_diff = hr_pt["f1"] - ib_pt["f1"]
            lines.append(
                f"| `{t}` | {sup} | {ib_pt['precision']:.3f} | {ib_pt['recall']:.3f} | {ib_pt['f1']:.3f} | "
                f"{hr_pt['precision']:.3f} | {hr_pt['recall']:.3f} | {hr_pt['f1']:.3f} | {f1_diff:+.3f} |"
            )

        lines.extend([
            "",
            "## 3. Ambiguous Homograph Disambiguation",
            "",
            "| Token | Count | IndicBERT Acc | HingRoBERTa Acc | Delta |",
            "|---|---|---|---|---|",
        ])

        for hg in sorted(ib_res["homographs"].keys()):
            ib_h = ib_res["homographs"][hg]
            hr_h = hr_res["homographs"][hg]
            delta = hr_h["accuracy"] - ib_h["accuracy"]
            lines.append(
                f"| `{hg}` | {ib_h['count']} | {ib_h['accuracy']:.1%} | {hr_h['accuracy']:.1%} | {delta:+.1%} |"
            )

        lines.extend([
            "",
            "## 4. Top Confusions (Gold -> Predicted)",
            "",
            "### IndicBERT v2 Top Confusions",
            "| Gold | Predicted | Count |",
            "|---|---|---|",
        ])
        for g, p, cnt in ib_res["top_confusions"]:
            lines.append(f"| `{g}` | `{p}` | {cnt} |")

        lines.extend([
            "",
            "### HingRoBERTa Top Confusions",
            "| Gold | Predicted | Count |",
            "|---|---|---|",
        ])
        for g, p, cnt in hr_res["top_confusions"]:
            lines.append(f"| `{g}` | `{p}` | {cnt} |")

    report_text = "\n".join(lines) + "\n"
    (out_dir / "indicbert_vs_hingroberta_report.md").write_text(report_text, encoding="utf-8")
    print(f"\n--> Benchmark complete! Saved reports to:")
    print(f"    - {out_dir / 'indicbert_vs_hingroberta_report.md'}")
    print(f"    - {out_dir / 'model_comparison.json'}")

    return benchmark_results


if __name__ == "__main__":
    run_benchmark()

