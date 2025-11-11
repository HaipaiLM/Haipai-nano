import argparse
import math
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from phase01.models.mor import MoRConfig, TinyMoR
from phase01.utils.logging import wandb_run
from phase01.utils.precision import autocast_precision, quantize_gradients


def stream_token_shards(shards: Sequence[Path], batch_size: int) -> Iterator[torch.Tensor]:
    while True:
        for shard in shards:
            data = torch.load(shard, map_location="cpu")
            inputs = data["input_ids"]
            labels = data["labels"]
            for i in range(0, inputs.size(0), batch_size):
                yield inputs[i : i + batch_size], labels[i : i + batch_size]


def eval_once(model: TinyMoR, shards: Sequence[Path], device, max_batches: int, batch_size: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        with tqdm(total=max_batches, desc="Eval", unit="batch") as pbar:
            batches_yielded = 0
            for shard in shards:
                data = torch.load(shard, map_location="cpu")
                inputs = data["input_ids"]
                labels = data["labels"]
                for i in range(0, inputs.size(0), batch_size):
                    input_ids = inputs[i : i + batch_size].to(device)
                    label_ids = labels[i : i + batch_size].to(device)
                    _, loss, _ = model(input_ids, labels=label_ids)
                    losses.append(loss.item())
                    batches_yielded += 1
                    pbar.update(1)
                    if batches_yielded >= max_batches:
                        break
                if batches_yielded >= max_batches:
                    break
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main():
    parser = argparse.ArgumentParser(description="Train tiny MoR (Phase 1) with progress bars + W&B.")
    parser.add_argument("--train_shards", nargs="+", required=True, type=Path)
    parser.add_argument("--val_shards", nargs="+", required=True, type=Path)
    parser.add_argument("--tokenizer_path", required=True)
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

    train_batches = stream_token_shards(args.train_shards, args.batch_size)

    def log_eval(step: int, run):
        val_loss = eval_once(
            model,
            args.val_shards,
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
        {"batch_size": args.batch_size},
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
