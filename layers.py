import copy
import math

import torch
import torch.nn as nn

Tensor = torch.Tensor
Module = nn.Module
ModuleList = nn.ModuleList


def clone(module: Module, N: int) -> ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_dim: int):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(vocab_dim, embed_dim))
        nn.init.uniform_(self.weight, -0.01, 0.01)
        self.scale = embed_dim**0.5

    def forward(self, x: Tensor, inverse: bool = False) -> Tensor:
        if inverse:
            return x @ nn.functional.normalize(self.weight, dim=-1).transpose(0, 1)
        return self.scale * nn.functional.normalize(self.weight[x], dim=-1)


class PositionalEncoding(nn.Module):
    enc: Tensor

    def __init__(self, embed_dim: int, dropout: float, max_length: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        enc = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000) / embed_dim))
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('enc', enc.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.enc[:, : x.size(1)])


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float):
        super(FeedForward, self).__init__()
        self.ff_1 = nn.Linear(embed_dim, ff_dim)
        self.ff_2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.ff_2(self.dropout(self.ff_1(x).relu()))


class ScaleNorm(nn.Module):
    def __init__(self, scale: float):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: Tensor, eps: float = 1e-5) -> Tensor:
        return self.scale * nn.functional.normalize(x, dim=-1, eps=eps)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.linears = clone(nn.Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

    def attention(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1) == 0, -torch.inf)
        return self.dropout(scores.softmax(dim=-1)) @ value

    def _reshape_from(self, x: Tensor) -> Tensor:
        return x.reshape(*x.size()[:2], self.num_heads, self.head_dim)

    def _reshape_to(self, x: Tensor) -> Tensor:
        return x.reshape(*x.size()[:2], -1)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        query, key, value = [
            self._reshape_from(linear(x)).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        outputs = self.attention(query, key, value, mask)
        return self.linears[-1](self._reshape_to(outputs.transpose(1, 2)))
