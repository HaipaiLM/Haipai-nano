import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

from phase01.data.curriculum import curriculum_schedule, score_dataset
from phase01.data.mga import MGAReformatter, iter_reformatted
from phase01.data.tokenizer_50k import train_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Phase 0 pipeline with explicit progress bars.")
    parser.add_argument("--dataset", required=True, help="HF dataset name or 'json' for local file.")
    parser.add_argument("--dataset_files", nargs="+", type=Path, help="Local JSONL file(s) when --dataset json.")
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--pacing", choices=["linear", "log"], default="linear")
    parser.add_argument("--total_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tokenizer_output", type=Path, default=Path("artifacts/tokenizer"))
    args = parser.parse_args()

    if args.dataset.lower() == "json":
        if not args.dataset_files:
            raise ValueError("--dataset_files is required when --dataset json")
        data_files = [str(path) for path in args.dataset_files]
        dataset = load_dataset("json", data_files=data_files, split="train", streaming=False)
    else:
        dataset = load_dataset(args.dataset, split="train", streaming=False)
    scores = score_dataset(dataset, text_key=args.text_key)
    schedule = curriculum_schedule(scores, args.total_steps, pacing=args.pacing)
    reformer = MGAReformatter(
        genres=["news", "analysis", "narrative", "dialog"],
        audiences=["experts", "students", "managers", "kids"],
        text_key=args.text_key,
    )
    augmented = []
    for example in iter_reformatted(dataset, reformer):
        augmented.append(example)
    print(f"Augmented {len(augmented)} samples via MGA.")
    with tqdm(total=1, desc="Tokenizer train", unit="phase") as pbar:
        train_tokenizer(args.dataset, args.text_key, args.tokenizer_output)
        pbar.update()


if __name__ == "__main__":
    main()
