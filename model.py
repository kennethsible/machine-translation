from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from layers import (
    Embedding,
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    ScaleNorm,
    clone,
)

Sublayer = Callable[[Tensor], Tensor]


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(embed_dim**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Sublayer) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 2)
        self.gate = nn.Parameter(torch.empty(1))
        nn.init.normal_(self.gate, mean=1.0, std=0.01)

    def forward(self, src_embs: Tensor, src_encs: Tensor, src_mask: Tensor | None = None) -> Tensor:
        src_encs = self.gate * src_embs + self.sublayers[0](
            src_encs, lambda x: self.self_attn(x, x, x, src_mask)
        )
        return src_embs + self.sublayers[1](src_encs, self.ff)


class Encoder(nn.Module):
    def __init__(
        self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float, num_layers: int
    ):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.norm = ScaleNorm(embed_dim**0.5)
        self.rgate = nn.Parameter(torch.empty(1))
        self.bgate = nn.Parameter(torch.empty(1))
        nn.init.normal_(self.rgate, mean=1.0, std=0.01)
        nn.init.normal_(self.bgate, mean=1.0, std=0.01)

    def forward(self, src_embs: Tensor, src_mask: Tensor | None = None) -> Tensor:
        src_encs = src_embs
        for layer in self.layers:
            src_encs = self.rgate * src_encs + self.bgate * layer(src_embs, src_encs, src_mask)
        return self.norm(src_encs)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.crss_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 3)
        self.gate = nn.Parameter(torch.empty(1))
        nn.init.normal_(self.gate, mean=1.0, std=0.01)

    def forward(
        self,
        tgt_embs: Tensor,
        src_encs: Tensor,
        tgt_encs: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        m = src_encs
        tgt_encs = self.gate * tgt_embs + self.sublayers[0](
            tgt_encs, lambda x: self.self_attn(x, x, x, tgt_mask)
        )
        tgt_encs = self.gate * tgt_embs + self.sublayers[1](
            tgt_encs, lambda x: self.crss_attn(x, m, m, src_mask)
        )
        return self.gate * tgt_embs + self.sublayers[2](tgt_encs, self.ff)


class Decoder(nn.Module):
    def __init__(
        self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float, num_layers: int
    ):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.norm = ScaleNorm(embed_dim**0.5)
        self.rgate = nn.Parameter(torch.empty(1))
        self.bgate = nn.Parameter(torch.empty(1))
        nn.init.normal_(self.rgate, mean=1.0, std=0.01)
        nn.init.normal_(self.bgate, mean=1.0, std=0.01)

    def forward(
        self,
        src_encs: Tensor,
        tgt_embs: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        tgt_encs = tgt_embs
        for layer in self.layers:
            tgt_encs = self.rgate * tgt_encs + self.bgate * layer(
                tgt_embs, src_encs, tgt_encs, src_mask, tgt_mask
            )
        return self.norm(tgt_encs)


class Model(nn.Module):
    def __init__(
        self,
        vocab_dim: int,
        embed_dim: int,
        ff_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
    ):
        super(Model, self).__init__()
        self.encoder = Encoder(embed_dim, ff_dim, num_heads, dropout, num_layers)
        self.decoder = Decoder(embed_dim, ff_dim, num_heads, dropout, num_layers)
        self.out_embed = Embedding(embed_dim, vocab_dim)
        self.src_embed = nn.Sequential(self.out_embed, PositionalEncoding(embed_dim, dropout))
        self.tgt_embed = nn.Sequential(self.out_embed, PositionalEncoding(embed_dim, dropout))

    def encode(self, src_nums: Tensor, src_mask: Tensor | None) -> Tensor:
        src_embs = self.src_embed(src_nums)
        return self.encoder(src_embs, src_mask)

    def decode(
        self,
        src_encs: Tensor,
        tgt_nums: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        tgt_embs = self.tgt_embed(tgt_nums)
        return self.decoder(src_encs, tgt_embs, src_mask, tgt_mask)

    def forward(
        self,
        src_nums: Tensor,
        tgt_nums: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        src_encs = self.encode(src_nums, src_mask)
        tgt_encs = self.decode(src_encs, tgt_nums, src_mask, tgt_mask)
        return self.out_embed(tgt_encs, inverse=True)
        return self.out_embed(tgt_encs, inverse=True)
