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
import gzip
import json
from pathlib import Path

import pyarrow.dataset as pa_ds
from tqdm.auto import tqdm

OUT_NAMES = {
    "fineweb": "fineweb_edu.jsonl",
    "c4": "c4_en.jsonl",
}


def iter_text_rows(folder: Path, limit: int, desc: str, batch_size: int, num_threads: int):
    """Stream text from local parquet or jsonl.gz files."""
    paths = list(folder.glob("*"))
    if not paths:
        raise FileNotFoundError(folder)
    ext = paths[0].suffix
    if ext == ".parquet":
        dataset = pa_ds.dataset(str(folder), format="parquet")
        scanner = dataset.scanner(
            columns=["text"],
            use_threads=num_threads > 1,
            batch_size=batch_size,
        )
        count = 0
        with tqdm(total=limit, desc=desc, unit="row") as pbar:
            for batch in scanner.to_batches():
                text_col = batch.column("text")
                for value in text_col:
                    if value is None:
                        continue
                    yield value.as_py()
                    count += 1
                    pbar.update(1)
                    if count >= limit:
                        return
    else:
        files = sorted(list(folder.glob("*.json")) + list(folder.glob("*.json.gz")))
        if not files:
            raise FileNotFoundError(f"No JSON/JSON.GZ files found in {folder}")
        count = 0
        with tqdm(total=limit, desc=desc, unit="row") as pbar:
            for file in files:
                opener = gzip.open if file.suffix == ".gz" else open
                with opener(file, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = record.get("text")
                        if not text:
                            continue
                        yield text
                        count += 1
                        pbar.update(1)
                        if count >= limit:
                            return


def write_jsonl(text_iter, path: Path, source: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text in text_iter:
            if not text:
                continue
            payload = {"source": source, "text": text}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser("Assemble local FineWeb + C4 into JSONL files.")
    parser.add_argument("--fineweb_dir", type=Path, required=True, help="Path to CC-MAIN-2025-26 folder.")
    parser.add_argument("--c4_dir", type=Path, required=True, help="Path to en/3.0.0/train folder.")
    parser.add_argument("--fineweb_limit", type=int, default=14_000_000)
    parser.add_argument("--c4_limit", type=int, default=10_000_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, default=Path("corpus"))
    args = parser.parse_args()

    fineweb_iter = iter_text_rows(
        args.fineweb_dir,
        args.fineweb_limit,
        "FineWeb CC-MAIN-2025-26",
        args.batch_size,
        args.num_threads,
    )
    write_jsonl(fineweb_iter, args.output_dir / OUT_NAMES["fineweb"], "fineweb-edu")

    c4_iter = iter_text_rows(
        args.c4_dir,
        args.c4_limit,
        "C4 en",
        args.batch_size,
        args.num_threads,
    )
    write_jsonl(c4_iter, args.output_dir / OUT_NAMES["c4"], "c4-en")


if __name__ == "__main__":
    main()
