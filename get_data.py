"""
get_data.py — download the Hindi-English POS data (LinCE `pos_hineng`) and save it
as the CSVs data_prep.py expects:

    pos_hineng_train.csv, pos_hineng_validation.csv, pos_hineng_test.csv

    python get_data.py

The Kaggle "pos_hineng" files in the blueprint are a copy of this same LinCE
dataset, so this replaces the manual Kaggle download. Needs `huggingface_hub`
and `pyarrow` (both come with transformers / pandas).

NOTE: LinCE hides the gold tags of its official test split (it's used for a
leaderboard), so the test CSV may have no tags. Use the validation CSV as
your test set — this script tells you which situation you're in.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

REPOS = ("lince-benchmark/lince", "lince")
REVISIONS = ("refs/convert/parquet", "main")


def find_files():
    from huggingface_hub import hf_hub_download, list_repo_files

    for repo in REPOS:
        for rev in REVISIONS:
            try:
                files = list_repo_files(repo, repo_type="dataset", revision=rev)
            except Exception as exc:
                print(f"  {repo}@{rev}: {type(exc).__name__}")
                continue
            hits = [f for f in files if "pos_hineng" in f and f.endswith((".parquet", ".csv", ".json", ".jsonl"))]
            if hits:
                print(f"  found {len(hits)} file(s) in {repo}@{rev}")
                return [(f, hf_hub_download(repo, f, repo_type="dataset", revision=rev)) for f in hits]
    return []


def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_json(path, lines=path.endswith(".jsonl"))


def split_name(remote: str) -> str | None:
    name = remote.lower()
    for split, keys in (("validation", ("validation", "valid", "dev")), ("train", ("train",)), ("test", ("test",))):
        if any(k in name for k in keys):
            return split
    return None


def main() -> None:
    print("Looking for LinCE pos_hineng on Hugging Face…")
    try:
        files = find_files()
    except ImportError:
        sys.exit("huggingface_hub is missing: pip install huggingface_hub pyarrow")
    if not files:
        sys.exit("Couldn't download automatically. Get the files by hand instead — see the README "
                 "section 'Getting the data'.")

    frames: dict[str, list[pd.DataFrame]] = {}
    for remote, local in files:
        split = split_name(remote)
        if split:
            frames.setdefault(split, []).append(read_any(local))

    for split, parts in frames.items():
        df = pd.concat(parts, ignore_index=True)
        cols = [c for c in ("words", "lid", "pos") if c in df.columns]
        out = pd.DataFrame({c: df[c].map(lambda v: json.dumps([str(x) for x in list(v)], ensure_ascii=False))
                            for c in cols})
        path = f"pos_hineng_{split}.csv"
        out.to_csv(path, index=False)
        labelled = df["pos"].map(lambda v: any(str(x).strip() for x in v)).sum() if "pos" in df else 0
        print(f"  wrote {path}: {len(df)} sentences, {labelled} with POS tags")

    print("\nNext step:")
    print("  python data_prep.py --train pos_hineng_train.csv --test pos_hineng_validation.csv")


if __name__ == "__main__":
    main()
