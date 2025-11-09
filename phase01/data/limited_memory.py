"""
Limited-Memory LM glue (arXiv:2505.15962).
We store factual triples externally and mask them from the LM loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import json
from tqdm.auto import tqdm


@dataclass
class FactStore:
    path: Path

    def save(self, entries: List[Dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for item in entries:
                f.write(json.dumps(item) + "\n")

    def load(self) -> List[Dict]:
        facts: List[Dict] = []
        if not self.path.exists():
            return facts
        with self.path.open() as f:
            for line in f:
                facts.append(json.loads(line))
        return facts


def extract_facts(dataset: Iterable[Dict], text_key: str = "text") -> List[Dict]:
    """Toy placeholder – in practice you'd run an IE system."""
    facts: List[Dict] = []
    for example in tqdm(dataset, desc="Extracting factual snippets", unit="sample"):
        text = example[text_key]
        snippet = text[:128]
        facts.append({"snippet": snippet})
    return facts


def mask_facts_in_labels(labels: List[int], fact_positions: List[Tuple[int, int]]) -> List[int]:
    masked = labels[:]
    for start, end in fact_positions:
        for idx in range(start, min(end, len(masked))):
            masked[idx] = -100
    return masked
