"""
Tokenize raw JSONL files into packed tensor shards so training never tokenizes on GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import multiprocessing as mp
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

_TOKENIZER = None


def _resolve_tokenizer_file(path: str) -> str:
    path = Path(path)
    if path.is_dir():
        file_path = path / "tokenizer.json"
    else:
        file_path = path
    if not file_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {file_path}")
    return str(file_path)


def _worker_init(tokenizer_path: str):
    global _TOKENIZER
    file_path = _resolve_tokenizer_file(tokenizer_path)
    _TOKENIZER = PreTrainedTokenizerFast(tokenizer_file=file_path)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token


def _encode_batch(text_batch: Sequence[str]):
    encoded = _TOKENIZER(list(text_batch), add_special_tokens=True, return_attention_mask=False)
    return encoded["input_ids"]


def iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text")
                if text:
                    yield text


def flush_shard(buffer_inputs: List[torch.Tensor], buffer_labels: List[torch.Tensor], shard_idx: int, output_dir: Path):
    if not buffer_inputs:
        return shard_idx
    input_batch = torch.stack(buffer_inputs)
    label_batch = torch.stack(buffer_labels)
    shard_path = output_dir / f"shard_{shard_idx:06d}.pt"
    torch.save({"input_ids": input_batch, "labels": label_batch}, shard_path)
    return shard_idx + 1


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser("Pre-tokenize JSONL files into fixed-length tensor shards.")
    parser.add_argument("--tokenizer_path", required=True, help="Path to dir or tokenizer.json file.")
    parser.add_argument("--jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--shard_size", type=int, default=1024, help="Number of sequences per shard.")
    parser.add_argument("--output_dir", type=Path, default=Path("tokenized"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encode_batch", type=int, default=256, help="Texts per worker batch.")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=_resolve_tokenizer_file(args.tokenizer_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_len = int(args.seq_len)

    token_buffer: List[int] = []
    shard_inputs: List[torch.Tensor] = []
    shard_labels: List[torch.Tensor] = []
    shard_idx = 0

    text_iter = iter_jsonl(args.jsonl)
    batches = batched(text_iter, args.encode_batch)

    if args.num_workers > 1:
        with mp.Pool(args.num_workers, initializer=_worker_init, initargs=(args.tokenizer_path,)) as pool:
            iterator = pool.imap(_encode_batch, batches)
            iterator = tqdm(iterator, desc="Tokenizing corpus", unit="batch")
            for token_lists in iterator:
                for ids in token_lists:
                    token_buffer.extend(ids + [eos_id])
                    while len(token_buffer) >= seq_len + 1:
                        input_ids = torch.tensor(token_buffer[:seq_len], dtype=torch.long)
                        labels = torch.tensor(token_buffer[1 : seq_len + 1], dtype=torch.long)
                        token_buffer = token_buffer[seq_len :]
                        shard_inputs.append(input_ids)
                        shard_labels.append(labels)

                        if len(shard_inputs) >= args.shard_size:
                            shard_idx = flush_shard(shard_inputs, shard_labels, shard_idx, args.output_dir)
                            shard_inputs.clear()
                            shard_labels.clear()
    else:
        _worker_init(args.tokenizer_path)
        for token_lists in tqdm(( _encode_batch(batch) for batch in batches), desc="Tokenizing corpus", unit="batch"):
            for ids in token_lists:
                token_buffer.extend(ids + [eos_id])
                while len(token_buffer) >= seq_len + 1:
                    input_ids = torch.tensor(token_buffer[:seq_len], dtype=torch.long)
                    labels = torch.tensor(token_buffer[1 : seq_len + 1], dtype=torch.long)
                    token_buffer = token_buffer[seq_len :]
                    shard_inputs.append(input_ids)
                    shard_labels.append(labels)

                    if len(shard_inputs) >= args.shard_size:
                        shard_idx = flush_shard(shard_inputs, shard_labels, shard_idx, args.output_dir)
                        shard_inputs.clear()
                        shard_labels.clear()

    if len(token_buffer) > seq_len:
        token_buffer.extend([eos_id] * seq_len)
        while len(token_buffer) >= seq_len + 1:
            input_ids = torch.tensor(token_buffer[:seq_len], dtype=torch.long)
            labels = torch.tensor(token_buffer[1 : seq_len + 1], dtype=torch.long)
            token_buffer = token_buffer[seq_len :]
            shard_inputs.append(input_ids)
            shard_labels.append(labels)

    if shard_inputs:
        shard_idx = flush_shard(shard_inputs, shard_labels, shard_idx, args.output_dir)


if __name__ == "__main__":
    main()
