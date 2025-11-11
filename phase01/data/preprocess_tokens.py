"""
Tokenize raw JSONL files into packed tensor shards so training never tokenizes on GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer


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


def main():
    parser = argparse.ArgumentParser("Pre-tokenize JSONL files into fixed-length tensor shards.")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--shard_size", type=int, default=1024, help="Number of sequences per shard.")
    parser.add_argument("--output_dir", type=Path, default=Path("tokenized"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    args.output_dir.mkdir(parents=True, exist_ok=True)

    token_buffer: List[int] = []
    shard_inputs: List[torch.Tensor] = []
    shard_labels: List[torch.Tensor] = []
    shard_idx = 0

    for text in tqdm(iter_jsonl(args.jsonl), desc="Tokenizing corpus", unit="sample"):
        encoded = tokenizer(text, add_special_tokens=True, return_attention_mask=False)
        ids = encoded["input_ids"] + [eos_id]
        token_buffer.extend(ids)

        while len(token_buffer) >= args.seq_len + 1:
            input_ids = torch.tensor(token_buffer[: args.seq_len], dtype=torch.long)
            labels = torch.tensor(token_buffer[1 : args.seq_len + 1], dtype=torch.long)
            token_buffer = token_buffer[args.seq_len :]
            shard_inputs.append(input_ids)
            shard_labels.append(labels)

            if len(shard_inputs) >= args.shard_size:
                shard_idx = flush_shard(shard_inputs, shard_labels, shard_idx, args.output_dir)
                shard_inputs.clear()
                shard_labels.clear()

    if len(token_buffer) > args.seq_len:
        token_buffer.extend([eos_id] * args.seq_len)
        while len(token_buffer) >= args.seq_len + 1:
            input_ids = torch.tensor(token_buffer[: args.seq_len], dtype=torch.long)
            labels = torch.tensor(token_buffer[1 : args.seq_len + 1], dtype=torch.long)
            token_buffer = token_buffer[args.seq_len :]
            shard_inputs.append(input_ids)
            shard_labels.append(labels)

    if shard_inputs:
        shard_idx = flush_shard(shard_inputs, shard_labels, shard_idx, args.output_dir)


if __name__ == "__main__":
    main()
