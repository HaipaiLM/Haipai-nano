import argparse
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from phase01.models.mor import MoRConfig, TinyMoR
from phase01.utils.logging import wandb_run
from phase01.utils.precision import autocast_precision, quantize_gradients


def collate_fn(batch, tokenizer, seq_len):
    texts = [ex["text"] for ex in batch]
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    return input_ids, labels


def stream_dataset(name: str, text_key: str):
    ds = load_dataset(name, split="train", streaming=True)
    for example in tqdm(ds, desc="Streaming dataset", unit="sample"):
        yield {"text": example[text_key]}


def main():
    parser = argparse.ArgumentParser(description="Train tiny MoR (Phase 1) with progress bars + W&B.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--project", default="haipai-phase1")
    parser.add_argument("--run_name", default="tiny-mor")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset_iter = stream_dataset(args.dataset, args.text_key)

    def batch_iterator():
        batch = []
        for example in dataset_iter:
            batch.append(example)
            if len(batch) == args.batch_size:
                yield collate_fn(batch, tokenizer, args.seq_len)
                batch = []

    model = TinyMoR(MoRConfig(vocab_size=tokenizer.vocab_size))
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    with wandb_run(args.project, args.run_name, {"seq_len": args.seq_len}, disable=args.disable_wandb) as run:
        progress = tqdm(total=args.steps, desc="Training MoR", unit="step")
        for input_ids, labels in batch_iterator():
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
            progress.set_postfix({"loss": loss.item(), "budget": aux["budget"], "token_keep": aux["token_keep"]})
            if run:
                run.log(
                    {
                        "loss": loss.item(),
                        "budget": aux["budget"],
                        "token_keep": aux["token_keep"],
                        "step": global_step,
                    }
                )
            if global_step >= args.steps:
                break
        progress.close()


if __name__ == "__main__":
    main()
