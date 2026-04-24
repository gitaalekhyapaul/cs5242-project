from __future__ import annotations

import math

import torch
from torch import nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SASRecBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = PointWiseFeedForward(hidden_size, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(
            query=normed,
            key=normed,
            value=normed,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = self.ffn(self.ffn_norm(x))
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_size: int = 128,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_size, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size
        self.clear_padding_item_embedding()

    def clear_padding_item_embedding(self) -> None:
        # nn.init.normal_(self.item_embedding.weight, std=0.02)
        # nn.init.normal_(self.position_embedding.weight, std=0.02)
        if self.item_embedding.padding_idx is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.item_embedding.padding_idx].fill_(0.0)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.item_embedding(input_ids) * math.sqrt(self.hidden_size)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        padding_mask = input_ids.eq(0)
        for block in self.blocks:
            x = block(x, padding_mask)
        return self.final_norm(x)

    def training_logits(
        self,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(input_ids)
        pos_emb = self.item_embedding(pos_ids)
        neg_emb = self.item_embedding(neg_ids)
        pos_logits = (encoded * pos_emb).sum(dim=-1)
        neg_logits = (encoded * neg_emb).sum(dim=-1)
        return pos_logits, neg_logits

    def score_candidates(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_repr = self.user_representation(input_ids)
        candidate_emb = self.item_embedding(candidate_ids)
        return torch.einsum("bd,bkd->bk", user_repr, candidate_emb)

    def user_representation(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(input_ids)
        lengths = input_ids.ne(0).sum(dim=1).clamp(min=1) - 1
        return encoded[torch.arange(encoded.size(0), device=input_ids.device), lengths]

    def score_all_items(self, input_ids: torch.Tensor) -> torch.Tensor:
        user_repr = self.user_representation(input_ids)
        all_item_emb = self.item_embedding.weight
        return user_repr @ all_item_emb.transpose(0, 1)
