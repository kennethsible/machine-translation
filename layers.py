import torch, torch.nn as nn
import math, copy

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Linear(nn.Module):

    def __init__(self, d_inp, d_out):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_inp))
        self.bias = nn.Parameter(torch.empty(d_out))
        nn.init.normal_(self.weight, std=0.01)
        nn.init.normal_(self.bias, std=0.01)

    def forward(self, x):
        return x @ self.weight.transpose(0, 1) + self.bias

class LogSoftmax(nn.Module):

    def __init__(self, embed_dim, vocab_size):
        super(LogSoftmax, self).__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        # https://aclanthology.org/N18-1031
        weight = nn.functional.normalize(self.weight, dim=-1)
        return torch.log_softmax(x @ weight.transpose(0, 1), dim=-1)

class Embedding(nn.Module):

    def __init__(self, embed_dim, vocab_size):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        # https://aclanthology.org/N18-1031
        return nn.functional.normalize(self.weight[x], dim=-1)

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        enc = torch.zeros(max_len, embed_dim, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000) / embed_dim))
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('enc', enc.unsqueeze(0))

    def forward(self, x):
        x = x + self.enc[:, : x.size(1)]
        return self.dropout(x)

class LayerNorm(nn.Module):

    def __init__(self, size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean, var = torch.mean(x, dim=-1, keepdim=True), torch.var(x, dim=-1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class ScaleNorm(nn.Module):
    # https://aclanthology.org/2019.iwslt-1.17

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(scale)
        self.eps = eps

    def forward(self, x):
        return self.scale * nn.functional.normalize(x, dim=-1, eps=self.eps)

class FeedForward(nn.Module):

    def __init__(self, embed_dim, ff_dim, dropout):
        super(FeedForward, self).__init__()
        self.ff_1 = Linear(embed_dim, ff_dim)
        self.ff_2 = Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.ff_2(self.dropout(self.ff_1(x).relu()))

def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -torch.inf)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ value

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.d_k = embed_dim // num_heads
        self.linears = clone(Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        num_batches = query.size(0)
        query, key, value = [linear(x).view(num_batches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))]
        x = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).reshape(num_batches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)
