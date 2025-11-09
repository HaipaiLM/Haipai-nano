from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        attn = attn.masked_fill(mask, float("-inf"))
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        probs = attn.softmax(dim=-1)
        out = probs @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class DynamicDepthRouter(nn.Module):
    """Router-Tuning MoD (arXiv:2410.13184)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, hidden: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.proj(hidden).squeeze(-1) / temperature
        probs = torch.sigmoid(logits)
        return probs


class ACMGate(nn.Module):
    """Adaptive Computation Module (arXiv:2312.10193)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(hidden))
        return hidden * gate


class SelfBudgeterHead(nn.Module):
    """Implements SelfBudgeter (arXiv:2505.11274) to modulate recursion depth per token."""

    def __init__(self, dim: int, max_budget: int):
        super().__init__()
        self.max_budget = max_budget
        self.proj = nn.Linear(dim, max_budget)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.proj(hidden)
        probs = logits.softmax(dim=-1)
        budget = probs.argmax(dim=-1)
        return budget, probs


class DTRRouter(nn.Module):
    """Dynamic token routing (arXiv:2509.00925 / LoT OpenReview) to skip low-importance tokens."""

    def __init__(self, dim: int, base_keep_ratio: float = 0.5, min_keep_ratio: float = 0.1):
        super().__init__()
        self.score_proj = nn.Linear(dim, 1)
        self.base_keep_ratio = base_keep_ratio
        self.min_keep_ratio = min_keep_ratio

    def forward(
        self,
        hidden: torch.Tensor,
        keep_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [B, T, C]
            keep_ratio: Optional [B] tensor specifying fraction of tokens to keep.
        Returns:
            token_mask: [B, T] bool mask for tokens that stay active.
            scores: [B, T] token scores.
        """
        scores = self.score_proj(hidden).squeeze(-1)
        B, T = scores.shape
        if keep_ratio is None:
            keep_ratio = torch.full((B,), self.base_keep_ratio, device=scores.device)
        keep_ratio = torch.clamp(keep_ratio, self.min_keep_ratio, 1.0)
        token_mask = torch.zeros(B, T, dtype=torch.bool, device=scores.device)
        for b in range(B):
            k = max(1, int(keep_ratio[b].item() * T))
            topk = torch.topk(scores[b], k=k).indices
            token_mask[b, topk] = True
        return token_mask, scores
