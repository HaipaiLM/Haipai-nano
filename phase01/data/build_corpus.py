"""
Corpus builder for Phase 0.

Usage example:
python phase01/data/build_corpus.py \
    --fineweb_limit 14000000 \
    --fineweb_snapshot CC-MAIN-2025-26 \
    --c4_limit 10000000 \
    --code_limit 200000 \
    --news_limit 200000

All loops stream with tqdm progress bars so you can watch throughput live.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm.auto import tqdm


def write_record(out_path: Path, source: str, text: str, extra: Optional[dict] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": source, "text": text}
    if extra:
        payload["meta"] = extra
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def gather_fineweb(snapshot: str, limit: int, out_path: Path):
    ds = load_dataset("HuggingFaceM4/FineWeb-Edu", split="train", streaming=True)
    count = 0
    for example in tqdm(ds, desc="FineWeb-Edu", unit="sample"):
        meta = example.get("meta", {})
        snap = meta.get("snapshot") or example.get("snapshot") or example.get("snapshot_id")
        if snapshot and snap != snapshot:
            continue
        text = example.get("text") or example.get("cleaned_text")
        if not text:
            continue
        write_record(out_path, "fineweb-edu", text, {"snapshot": snap})
        count += 1
        if count >= limit:
            break


def gather_c4(limit: int, out_path: Path):
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    count = 0
    for example in tqdm(ds, desc="C4-en", unit="sample"):
        text = example.get("text")
        if not text:
            continue
        write_record(out_path, "c4-en", text)
        count += 1
        if count >= limit:
            break


def gather_code(limit: int, out_path: Path):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    count = 0
    for example in tqdm(ds, desc="Code/Reasoning", unit="sample"):
        question = example.get("question", "")
        answer = example.get("answer", "")
        text = f"Q: {question}\nA: {answer}"
        write_record(out_path, "gsm8k", text)
        count += 1
        if count >= limit:
            break


def gather_news(limit: int, out_path: Path):
    ds = load_dataset("ag_news", split="train")
    count = 0
    for example in tqdm(ds, desc="News", unit="sample"):
        text = f"{example.get('title', '')}\n{example.get('description', '')}"
        write_record(out_path, "ag_news", text, {"label": int(example.get("label", -1))})
        count += 1
        if count >= limit:
            break


def main():
    parser = argparse.ArgumentParser(description="Assemble curated corpus slices with progress bars.")
    parser.add_argument("--output_dir", type=Path, default=Path("corpus"))
    parser.add_argument("--fineweb_snapshot", default="CC-MAIN-2025-26")
    parser.add_argument("--fineweb_limit", type=int, default=14_000_000)
    parser.add_argument("--c4_limit", type=int, default=10_000_000)
    parser.add_argument("--code_limit", type=int, default=200_000)
    parser.add_argument("--news_limit", type=int, default=200_000)
    args = parser.parse_args()

    out_fineweb = args.output_dir / "fineweb_edu.jsonl"
    out_c4 = args.output_dir / "c4_en.jsonl"
    out_code = args.output_dir / "code_reason.jsonl"
    out_news = args.output_dir / "news.jsonl"

    gather_fineweb(args.fineweb_snapshot, args.fineweb_limit, out_fineweb)
    gather_c4(args.c4_limit, out_c4)
    gather_code(args.code_limit, out_code)
    gather_news(args.news_limit, out_news)


if __name__ == "__main__":
    main()
