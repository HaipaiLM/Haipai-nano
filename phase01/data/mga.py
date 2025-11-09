"""
Massive Genre-Audience reformulation layer (arXiv:2502.04235).
Wraps any dataset iterator and prepends genre/audience prompts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence

from tqdm.auto import tqdm


@dataclass
class MGAReformatter:
    genres: Sequence[str]
    audiences: Sequence[str]
    text_key: str = "text"

    def __call__(self, example: Dict) -> Dict:
        prefix = f"[Genre: {random.choice(self.genres)}] [Audience: {random.choice(self.audiences)}] "
        example = dict(example)
        example[self.text_key] = prefix + example[self.text_key]
        return example


def iter_reformatted(dataset: Iterable[Dict], reformer: MGAReformatter) -> Iterator[Dict]:
    for example in tqdm(dataset, desc="Applying MGA reformulation", unit="sample"):
        yield reformer(example)
