"""
Offline tokenizer -> shard converter with multiprocessing and progress bars.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

TOKENIZER: PreTrainedTokenizerFast | None = None


def resolve_tokenizer_file(path: str) -> str:
    p = Path(path)
    file_path = p / "tokenizer.json" if p.is_dir() else p
    if not file_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {file_path}")
    return str(file_path)


def worker_init(tokenizer_path: str):
    global TOKENIZER
    TOKENIZER = PreTrainedTokenizerFast(tokenizer_file=resolve_tokenizer_file(tokenizer_path))
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token


def encode_batch(text_batch: Sequence[str]) -> List[List[int]]:
    encoded = TOKENIZER(list(text_batch), add_special_tokens=True, return_attention_mask=False)
    return encoded["input_ids"]


def iter_jsonl(files: Iterable[Path]) -> Iterator[str]:
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text")
                if text:
                    yield text


def main():
    parser = argparse.ArgumentParser(description="Tokenize JSONL corpus into tensor shards.")
    parser.add_argument("--tokenizer_path", required=True, help="Directory or tokenizer.json file.")
    parser.add_argument("--jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--shard_size", type=int, default=1024)
    parser.add_argument("--output_dir", type=Path, default=Path("tokenized"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_texts", type=int, default=256)
    args = parser.parse_args()

    def _coerce_seq_len(v, default=4096):
        try:
            if isinstance(v, int) and v > 0:
                return v
            if isinstance(v, str):
                v = v.strip()
                if v.lower() == "none" or v == "":
                    return default
                iv = int(v)
                return iv if iv > 0 else default
        except Exception:
            pass
        return default

    seq_len = _coerce_seq_len(getattr(args,'seq_len', None))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_tokenizer = PreTrainedTokenizerFast(tokenizer_file=resolve_tokenizer_file(args.tokenizer_path))
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    eos_id = base_tokenizer.eos_token_id

    buffer: List[int] = []
    shard_inputs: List[torch.Tensor] = []
    shard_labels: List[torch.Tensor] = []
    shard_idx = 0

    def flush():
        nonlocal shard_idx
        if shard_inputs:
            shard_path = output_dir / f"shard_{shard_idx:06d}.pt"
            torch.save({"input_ids": torch.stack(shard_inputs), "labels": torch.stack(shard_labels)}, shard_path)
            shard_idx += 1
            shard_inputs.clear()
            shard_labels.clear()

    text_iter = iter_jsonl(args.jsonl)

    def batched(iterator, size):
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    batches = batched(text_iter, args.batch_texts)

    if args.num_workers > 1:
        with mp.Pool(args.num_workers, initializer=worker_init, initargs=(args.tokenizer_path,)) as pool:
            for token_lists in tqdm(pool.imap(encode_batch, batches), desc="Tokenizing", unit="batch"):
                for ids in token_lists:
                    buffer.extend(ids + [eos_id])
                    while len(buffer) >= seq_len + 1:
                        inp = torch.tensor(buffer[:seq_len], dtype=torch.long)
                        lbl = torch.tensor(buffer[1 : seq_len + 1], dtype=torch.long)
                        shard_inputs.append(inp)
                        shard_labels.append(lbl)
                        buffer = buffer[seq_len:]
                        if len(shard_inputs) >= args.shard_size:
                            flush()
    else:
        worker_init(args.tokenizer_path)
        for token_lists in tqdm((encode_batch(batch) for batch in batches), desc="Tokenizing", unit="batch"):
            for ids in token_lists:
                buffer.extend(ids + [eos_id])
                while len(buffer) >= seq_len + 1:
                    inp = torch.tensor(buffer[:seq_len], dtype=torch.long)
                    lbl = torch.tensor(buffer[1 : seq_len + 1], dtype=torch.long)
                    shard_inputs.append(inp)
                    shard_labels.append(lbl)
                    buffer = buffer[seq_len:]
                    if len(shard_inputs) >= args.shard_size:
                        flush()

    if len(buffer) > seq_len:
        buffer.extend([eos_id] * seq_len)
        while len(buffer) >= seq_len + 1:
            inp = torch.tensor(buffer[:seq_len], dtype=torch.long)
            lbl = torch.tensor(buffer[1 : seq_len + 1], dtype=torch.long)
            shard_inputs.append(inp)
            shard_labels.append(lbl)
            buffer = buffer[seq_len:]
    flush()


if __name__ == "__main__":
    main()
