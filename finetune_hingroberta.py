"""
finetune_hingroberta.py — Fine-tune HingRoBERTa (l3cube-pune/hing-roberta)
for Hinglish POS Tagging (17 Universal Dependencies tags).

Preserves existing IndicBERT checkpoint intact in models/indicbert_pos.
Saves best HingRoBERTa checkpoint in models/hingroberta_pos.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from metrics import score
from tagset import UD_TAGS, normalize_tag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "l3cube-pune/hing-roberta"
OUTPUT_DIR = Path("models/hingroberta_pos")

TAG2ID = {tag: i for i, tag in enumerate(UD_TAGS)}
ID2TAG = {i: tag for i, tag in enumerate(UD_TAGS)}


class HinglishPOSDataset(Dataset):
    def __init__(self, data_path: str | Path, tokenizer, max_len: int = 128):
        raw_data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.samples = []

        for sentence in raw_data:
            words = [x["token"] for x in sentence]
            tags = [normalize_tag(x["tag"]) for x in sentence]

            tokenized = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
            )

            word_ids = tokenized.word_ids(0)
            labels = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                elif wid != prev_wid:
                    labels.append(TAG2ID.get(tags[wid], TAG2ID["X"]))
                else:
                    labels.append(-100)
                prev_wid = wid

            self.samples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            diff = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * diff)
            attention_mask.append(x["attention_mask"] + [0] * diff)
            labels.append(x["labels"] + [self.label_pad_id] * diff)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def evaluate_dataset(model, dataset: HinglishPOSDataset, collator, device, batch_size: int = 16) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    
    all_preds = []
    all_gold = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            mask = labels != -100
            for p, l, m in zip(preds, labels, mask):
                valid_p = p[m].cpu().tolist()
                valid_l = l[m].cpu().tolist()
                for pred_id, gold_id in zip(valid_p, valid_l):
                    all_preds.append(ID2TAG[pred_id])
                    all_gold.append(ID2TAG[gold_id])

    return score(all_gold, all_preds)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HingRoBERTa on Hinglish POS data")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for scheduler")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)
    output_path = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}", flush=True)

    print(f"--> Loading tokenizer and model: {MODEL_NAME}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(UD_TAGS),
            id2label=ID2TAG,
            label2id=TAG2ID,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        ).to(device)
    except Exception:
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(UD_TAGS),
            id2label=ID2TAG,
            label2id=TAG2ID,
            ignore_mismatched_sizes=True,
        ).to(device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collator = DynamicPaddingCollator(pad_token_id=pad_token_id)

    print("--> Preparing datasets...", flush=True)
    train_dataset = HinglishPOSDataset("data/train.json", tokenizer)
    dev_dataset = HinglishPOSDataset("data/dev.json", tokenizer)
    test_dataset = HinglishPOSDataset("data/test.json", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print(f"--> Fine-tuning HingRoBERTa for {args.epochs} epochs ({len(train_loader)} batches/epoch with dynamic padding)...", flush=True)
    print(f"    Total steps: {total_training_steps}, Warmup steps: {warmup_steps}, LR: {args.lr}", flush=True)

    best_dev_macro_f1 = 0.0
    best_dev_metrics = None
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(f"    Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)
        dev_metrics = evaluate_dataset(model, dev_dataset, collator, device, batch_size=args.batch_size)
        epoch_duration = time.time() - epoch_start

        print(f"==> Epoch {epoch} Complete ({epoch_duration:.1f}s) | Avg Loss: {avg_loss:.4f} | "
              f"Dev Acc: {dev_metrics['accuracy']:.4f} | Dev Macro-F1: {dev_metrics['macro_f1']:.4f} | "
              f"Dev Weighted-F1: {dev_metrics['weighted_f1']:.4f}", flush=True)

        # Selection criterion: validation Macro-F1 (per user instruction)
        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_dev_metrics = dev_metrics
            best_epoch = epoch
            print(f"    [*] New best validation Macro-F1: {best_dev_macro_f1:.4f}! Saving checkpoint to {output_path}...", flush=True)
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

    total_duration = time.time() - start_time
    print(f"\n--> Training Complete in {total_duration:.1f}s!", flush=True)
    print(f"--> Best Epoch: {best_epoch} with Dev Macro-F1: {best_dev_metrics['macro_f1']:.4f}, Dev Acc: {best_dev_metrics['accuracy']:.4f}", flush=True)

    # Evaluate best model on test set
    print(f"--> Loading best model from {output_path} to evaluate on test set...", flush=True)
    best_model = AutoModelForTokenClassification.from_pretrained(output_path).to(device)
    test_metrics = evaluate_dataset(best_model, test_dataset, collator, device, batch_size=args.batch_size)

    print(f"\n========================================================", flush=True)
    print(f"HingRoBERTa Final Test Set Results:", flush=True)
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}", flush=True)
    print(f"  Macro-F1:    {test_metrics['macro_f1']:.4f}", flush=True)
    print(f"  Weighted-F1: {test_metrics['weighted_f1']:.4f}", flush=True)
    print(f"========================================================", flush=True)

    # Save training summary metadata
    summary = {
        "model_name": MODEL_NAME,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_epoch": best_epoch,
        "training_time_seconds": total_duration,
        "dev_metrics": {k: v for k, v in best_dev_metrics.items() if k != "per_tag"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "per_tag"},
        "test_per_tag": test_metrics["per_tag"],
    }
    (output_path / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"--> Saved training summary to {output_path / 'training_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
