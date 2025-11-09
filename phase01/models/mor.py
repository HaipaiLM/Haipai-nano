from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    ACMGate,
    Attention,
    DTRRouter,
    DynamicDepthRouter,
    RMSNorm,
    SelfBudgeterHead,
    SwiGLU,
)


@dataclass
class MoRConfig:
    vocab_size: int = 50_000
    d_model: int = 768
    n_layer: int = 8
    n_head: int = 12
    d_ff: int = 2048
    max_seq_len: int = 2048
    max_recursions: int = 3
    router_temperature: float = 1.0
    dropout: float = 0.1


class MoRBlock(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.attn = Attention(config.d_model, config.n_head)
        self.mlp = SwiGLU(config.d_model, config.d_ff)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.mod_router = DynamicDepthRouter(config.d_model)
        self.acm_gate = ACMGate(config.d_model)
        self.token_router = DTRRouter(config.d_model, base_keep_ratio=0.6)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        temperature: float,
        keep_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth_probs = self.mod_router(hidden, temperature)
        active = (depth_probs > 0.5).unsqueeze(-1)
        normed = self.norm1(hidden)
        token_mask, token_scores = self.token_router(normed, keep_ratio)
        attn_out = self.attn(normed, attn_mask=attn_mask)
        attn_out = attn_out * token_mask.unsqueeze(-1)
        hidden = hidden + self.dropout(attn_out) * active
        gated = self.acm_gate(self.norm2(hidden))
        hidden = hidden + self.dropout(self.mlp(gated)) * active
        return hidden, depth_probs, token_mask


class TinyMoR(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.blocks = nn.ModuleList(MoRBlock(config) for _ in range(config.n_layer))
        self.router = nn.Linear(config.d_model, config.max_recursions)
        self.budget_head = SelfBudgeterHead(config.d_model, config.max_recursions)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError("Sequence length exceeds model capacity.")

        pos = self.pos_embed[:, :T, :]
        hidden = self.embed(input_ids) + pos
        budget, budget_probs = self.budget_head(hidden)
        budget_ratio = budget.float().mean(dim=1) / max(1, self.config.max_recursions)
        depth_logits = self.router(hidden) / self.config.router_temperature
        depth = depth_logits.softmax(dim=-1).argmax(dim=-1)
        attn_mask = attention_mask if attention_mask is not None else torch.ones(B, T, dtype=torch.bool, device=hidden.device)

        router_losses = []
        token_stats = []
        for recursion in range(self.config.max_recursions):
            active = depth >= recursion
            if not active.any():
                continue
            mask = attn_mask & active.unsqueeze(-1)
            keep_ratio = budget_ratio
            for block in self.blocks:
                hidden, probs, token_mask = block(hidden, mask, self.config.router_temperature, keep_ratio)
                router_losses.append((probs.mean() - 0.5).abs())
                token_stats.append(token_mask.float().mean())

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            aux = torch.stack(router_losses).mean() if router_losses else torch.tensor(0.0, device=hidden.device)
            budget_reg = (budget.float().mean() - (self.config.max_recursions / 2)) ** 2
            token_reg = torch.stack(token_stats).mean() if token_stats else torch.tensor(0.0, device=hidden.device)
            loss = ce + 0.01 * aux + 0.01 * budget_reg + 0.01 * token_reg
        stats = {
            "budget": budget_probs.mean().item(),
            "token_keep": torch.stack(token_stats).mean().item() if token_stats else 0.0,
        }
        return logits, loss, stats
