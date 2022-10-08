from manager import clone
import math, torch, torch.nn as nn

class Linear(nn.Module):

    def __init__(self, d_inp, d_out):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(d_out, d_inp))
        self.bias = torch.nn.Parameter(torch.empty(d_out))
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.normal_(self.bias, std=0.01)

    def forward(self, x):
        return x @ self.weight.transpose(0, 1) + self.bias

class LogSoftmax(nn.Module):

    def __init__(self, d_model, vocab):
        super(LogSoftmax, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab, d_model))
        torch.nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        # https://aclanthology.org/N18-1031
        weight = nn.functional.normalize(self.weight, dim=-1)
        return torch.log_softmax(x @ weight.transpose(0, 1), dim=-1)

class Embedding(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab, d_model))
        torch.nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        # https://aclanthology.org/N18-1031
        return nn.functional.normalize(self.weight[x], dim=-1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        enc = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('enc', enc.unsqueeze(0))

    def forward(self, x):
        x = x + self.enc[:, : x.size(1)]
        return self.dropout(x)

class LabelSmoothing(nn.Module):

    def __init__(self, smoothing):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        l = torch.gather(input, dim=-1, index=target.unsqueeze(-1))
        return (self.smoothing - 1) * l.sum() - self.smoothing * input.mean()

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

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ff_1 = Linear(d_model, d_ff)
        self.ff_2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.ff_2(self.dropout(self.ff_1(x).relu()))

def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None: # TODO Score Caching
            scores = scores.masked_fill(mask == 0, -torch.inf)
        scores = torch.softmax(scores, dim=-1)
        if dropout: scores = dropout(scores)
        return scores @ value

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clone(Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        query, key, value = [
            linear(x).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        x = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).reshape(n_batches, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)
