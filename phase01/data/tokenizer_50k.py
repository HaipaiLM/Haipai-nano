import argparse
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm.auto import tqdm


def train_tokenizer(dataset: str, text_key: str, output_dir: Path, vocab_size: int = 50_000) -> None:
    ds = load_dataset(dataset, split="train", streaming=True)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )

    def iterator():
        for example in tqdm(ds, desc="Tokenizer samples", unit="sample"):
            yield example[text_key]

    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=None)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    print(f"Saved tokenizer to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train 50K-token tokenizer with visible progress.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--vocab_size", type=int, default=50_000)
    args = parser.parse_args()
    train_tokenizer(args.dataset, args.text_key, args.output_dir, args.vocab_size)


if __name__ == "__main__":
    main()
