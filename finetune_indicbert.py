"""
finetune_indicbert.py — Robust IndicBERT v2 fine-tuning on Hinglish POS data.
Direct PyTorch training loop (no 'accelerate' dependency, CPU-friendly).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from tagset import UD_TAGS, normalize_tag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CACHE_DIR = Path(os.path.expanduser("~/.cache/huggingface/hub"))
INDIC_POS = CACHE_DIR / "models--ai4bharat--IndicBERTv2-alpha-POS-tagging"
MODEL_NAME = "ai4bharat/IndicBERTv2-alpha-POS-tagging" if INDIC_POS.exists() else "ai4bharat/IndicBERTv2-MLM-only"
OUTPUT_DIR = Path("models/indicbert_pos")

TAG2ID = {tag: i for i, tag in enumerate(UD_TAGS)}
ID2TAG = {i: tag for i, tag in enumerate(UD_TAGS)}


class POSDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 128):
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
                padding="max_length",
                return_tensors="pt",
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
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            mask = labels != -100
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    return (correct / total) if total > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    print(f"--> Loading tokenizer and model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(UD_TAGS),
            id2label=ID2TAG,
            label2id=TAG2ID,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        ).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(UD_TAGS),
            id2label=ID2TAG,
            label2id=TAG2ID,
            ignore_mismatched_sizes=True,
        ).to(device)

    print("--> Preparing datasets...")
    train_dataset = POSDataset("data/train.json", tokenizer)
    dev_dataset = POSDataset("data/dev.json", tokenizer)
    test_dataset = POSDataset("data/test.json", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    epochs = 3

    print(f"--> Starting fine-tuning for {epochs} epochs ({len(train_loader)} batches/epoch)...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
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
            optimizer.step()

            total_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(f"    Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        dev_acc = evaluate(model, dev_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"==> Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Dev Accuracy: {dev_acc:.4f}")

        if dev_acc > best_acc:
            best_acc = dev_acc
            print(f"    Saving best model so far to {OUTPUT_DIR}...")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    test_acc = evaluate(model, test_loader, device)
    print(f"\n--> Success! Best Dev Accuracy: {best_acc:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"--> Fine-tuned model is saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
