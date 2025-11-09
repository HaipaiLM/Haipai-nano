"""
Curriculum + pacing utilities inspired by arXiv:2506.11300.
Every loop surfaces a tqdm progress bar so the user always sees progress.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm

DifficultyFn = Callable[[Dict], float]


def build_default_metrics(text_key: str = "text") -> List[DifficultyFn]:
    def token_length(example: Dict) -> float:
        return len(example[text_key].split())

    def punctuation_density(example: Dict) -> float:
        text = example[text_key]
        punct = sum(text.count(p) for p in ",.;:?!")
        return punct / max(1, len(text))

    def unique_ratio(example: Dict) -> float:
        text = example[text_key]
        chars = len(text)
        unique = len(set(text))
        return unique / max(1, chars)

    return [token_length, punctuation_density, unique_ratio]


def score_dataset(dataset: Dataset, metrics: Optional[Sequence[DifficultyFn]] = None, text_key: str = "text") -> List[float]:
    metrics = list(metrics) if metrics else build_default_metrics(text_key)
    scores: List[float] = []
    for example in tqdm(dataset, desc="Scoring samples (curriculum)", unit="sample"):
        vals = [fn(example) for fn in metrics]
        scores.append(sum(vals) / len(vals))
    return scores


def curriculum_schedule(scores: Sequence[float], total_steps: int, pacing: str = "linear") -> List[int]:
    sorted_idx = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1])]
    schedule: List[int] = []
    n = len(sorted_idx)
    for step in tqdm(range(total_steps), desc="Building curriculum schedule", unit="step"):
        if pacing == "linear":
            frac = (step + 1) / total_steps
        elif pacing == "log":
            frac = math.log(step + 2, total_steps + 1)
        else:
            raise ValueError(f"Unknown pacing strategy: {pacing}")
        upto = max(1, int(frac * n))
        schedule.append(random.choice(sorted_idx[:upto]))
    return schedule


@dataclass
class CurriculumBatcher:
    dataset: Dataset
    schedule: Sequence[int]
    batch_size: int

    def __iter__(self) -> Iterable[List[Dict]]:
        batch: List[Dict] = []
        for idx in tqdm(self.schedule, desc="Sampling curriculum batches", unit="step"):
            batch.append(self.dataset[int(idx)])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
