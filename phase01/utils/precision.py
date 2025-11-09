"""
Low-precision helpers (arXiv:2505.01043 inspired).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import torch


@contextmanager
def autocast_precision(device_type: str) -> Iterator[None]:
    if device_type == "cuda" and torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            yield
    else:
        yield


def quantize_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.data.to(torch.float32)
        grad = torch.round(grad * 128) / 128  # 1/128 ≈ 8-bit resolution
        param.grad.data.copy_(grad)
