import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from phase01.models.mor import MoRConfig, TinyMoR
from phase01.utils.logging import wandb_run
from phase01.utils.precision import autocast_precision, quantize_gradients


def collate_fn(texts: Sequence[str], tokenizer, seq_len: int):
    encoded = tokenizer(list(texts), padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    return input_ids, labels


def cycle_jsonl(files: Sequence[Path]) -> Iterator[str]:
    while True:
        for path in files:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    yield line


def stream_batches(files: Sequence[Path], batch_size: int, tokenizer, seq_len: int):
    iterator = cycle_jsonl(files)
    batch_texts: List[str] = []
    for line in iterator:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = record.get("text")
        if not text:
            continue
        batch_texts.append(text)
        if len(batch_texts) == batch_size:
            yield collate_fn(batch_texts, tokenizer, seq_len)
            batch_texts = []


def eval_once(model: TinyMoR, files: Sequence[Path], tokenizer, seq_len: int, device, max_batches: int, batch_size: int) -> float:
    model.eval()
    losses = []
    iterator = cycle_jsonl(files)
    batch_texts: List[str] = []
    with torch.no_grad():
        with tqdm(total=max_batches, desc="Eval", unit="batch") as pbar:
            for line in iterator:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text")
                if not text:
                    continue
                batch_texts.append(text)
                if len(batch_texts) == batch_size:
                    input_ids, labels = collate_fn(batch_texts, tokenizer, seq_len)
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    _, loss, _ = model(input_ids, labels=labels)
                    losses.append(loss.item())
                    pbar.update(1)
                    batch_texts = []
                    if pbar.n >= max_batches:
                        break
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main():
    parser = argparse.ArgumentParser(description="Train tiny MoR (Phase 1) with progress bars + W&B.")
    parser.add_argument("--train_files", nargs="+", required=True, type=Path)
    parser.add_argument("--val_files", nargs="+", required=True, type=Path)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--project", default="haipai-phase1")
    parser.add_argument("--run_name", default="tiny-mor")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = TinyMoR(MoRConfig(vocab_size=tokenizer.vocab_size))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_batches = stream_batches(args.train_files, args.batch_size, tokenizer, args.seq_len)

    def log_eval(step: int, run):
        val_loss = eval_once(
            model,
            args.val_files,
            tokenizer,
            args.seq_len,
            device,
            args.eval_batches,
            args.batch_size,
        )
        ppl = math.exp(min(20, val_loss))
        if run:
            run.log({"val_loss": val_loss, "val_ppl": ppl, "step": step})
        print(f"[Eval] step {step}: loss={val_loss:.4f} | ppl={ppl:.2f}")

    global_step = 0
    with wandb_run(
        args.project,
        args.run_name,
        {"seq_len": args.seq_len, "batch_size": args.batch_size},
        disable=args.disable_wandb,
    ) as run:
        progress = tqdm(total=args.steps, desc="Training MoR", unit="step")
        for input_ids, labels in train_batches:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast_precision(device.type):
                logits, loss, aux = model(input_ids, labels=labels)
            loss.backward()
            quantize_gradients(model.parameters())
            optimizer.step()

            global_step += 1
            progress.update(1)
            progress.set_postfix(
                {
                    "loss": loss.item(),
                    "budget": aux["budget"],
                    "token_keep": aux["token_keep"],
                }
            )
            if run:
                run.log(
                    {
                        "loss": loss.item(),
                        "budget": aux["budget"],
                        "token_keep": aux["token_keep"],
                        "step": global_step,
                    }
                )
            if global_step % args.eval_steps == 0:
                log_eval(global_step, run)
            if global_step >= args.steps:
                break
        progress.close()


if __name__ == "__main__":
    main()
