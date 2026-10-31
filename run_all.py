"""
run_all.py — the whole project in one command.

    python run_all.py

Does every step in order, skipping any step whose output already exists:
  1. Download the dataset (LinCE Hindi-English POS)      -> pos_hineng_*.csv
  2. Clean it into train / dev / test JSON                -> data/*.json
  3. Train: calibrate rules, train HMM, tune the arbiter  -> models/*
  4. Evaluate on the test set                             -> results/report.md
  5. Open the web app in your browser

Options:
    --no-transformer   skip the BERT model (faster; rules + HMM only)
    --force            redo every step even if its output exists
    --no-app           stop after evaluation
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
PY = sys.executable


def step(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


def run(*args: str) -> None:
    result = subprocess.run([PY, *args], cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"\nStopped: `python {' '.join(args)}` failed (see the error above).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-app", action="store_true")
    args = ap.parse_args()

    has_torch = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))
    use_trf = has_torch and not args.no_transformer
    if not has_torch and not args.no_transformer:
        print("torch/transformers not installed — running with rules + HMM only.")
    trf_flag = [] if use_trf else ["--no-transformer"]

    step("1/5  Dataset")
    if args.force or not (HERE / "pos_hineng_train.csv").exists():
        run("get_data.py")
    else:
        print("pos_hineng_train.csv already here — skipping download.")
    test_csv = "pos_hineng_validation.csv" if (HERE / "pos_hineng_validation.csv").exists() else "pos_hineng_test.csv"

    step("2/5  Cleaning data")
    if args.force or not (HERE / "data" / "train.json").exists():
        run("data_prep.py", "--train", "pos_hineng_train.csv", "--test", test_csv)
    else:
        print("data/train.json already exists — skipping.")

    step("3/5  Training" + (" (first run downloads the transformer, ~700 MB)" if use_trf else ""))
    if args.force or not (HERE / "models" / "hmm.npz").exists():
        run("train.py", *trf_flag)
    else:
        print("models/ already built — skipping. Use --force to retrain.")

    step("4/5  Evaluating on the test set")
    run("evaluate.py", *trf_flag)

    if args.no_app:
        return
    step("5/5  Starting the web app — press Ctrl+C here to stop it")
    subprocess.run([PY, "-m", "streamlit", "run", "app.py"], cwd=HERE)


if __name__ == "__main__":
    main()
