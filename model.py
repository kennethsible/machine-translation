from layers import clone, Embedding, PositionalEncoding, \
    LogSoftmax, ScaleNorm, FeedForward, MultiHeadAttention
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping:

    def __init__(self, patience, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0

    def __call__(self, curr_loss, prev_loss):
        if (curr_loss - prev_loss) > self.min_delta:
            self.count += 1
            if self.count >= self.patience:
                return True

class SublayerConnection(nn.Module):

    def __init__(self, scale, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(scale)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(n_heads, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        scale = torch.tensor(d_model, dtype=torch.float32)
        self.sublayers = clone(SublayerConnection(torch.sqrt(scale), dropout), 2)

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_att(x, x, x, mask))
        return self.sublayers[1](x, self.ff)

class Encoder(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, dropout, N):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(d_model, d_ff, n_heads, dropout), N)
        scale = torch.tensor(d_model, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_att  = MultiHeadAttention(n_heads, d_model)
        self.cross_att = MultiHeadAttention(n_heads, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        scale = torch.tensor(d_model, dtype=torch.float32)
        self.sublayers = clone(SublayerConnection(torch.sqrt(scale), dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_att(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.cross_att(x, m, m, src_mask))
        return self.sublayers[2](x, self.ff)

class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, dropout, N):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(d_model, d_ff, n_heads, dropout), N)
        scale = torch.tensor(d_model, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Model(nn.Module):

    def __init__(self, vocab_size, d_model=512, d_ff=2048, n_heads=8, dropout=0.1, N=6):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(d_model, d_ff, n_heads, dropout, N)
        self.decoder = Decoder(d_model, d_ff, n_heads, dropout, N)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.src_embed = nn.Sequential(Embedding(d_model, vocab_size), PositionalEncoding(d_model, dropout))
        self.tgt_embed = nn.Sequential(Embedding(d_model, vocab_size), PositionalEncoding(d_model, dropout))
        self.generator = LogSoftmax(d_model, vocab_size)

    def forward(self, src_nums, tgt_nums, src_mask, tgt_mask):
        return self.decode(self.encode(src_nums, src_mask), src_mask, tgt_nums, tgt_mask)

    def encode(self, src_nums, src_mask):
        return self.encoder(self.src_embed(src_nums), src_mask)

    def decode(self, memory, src_mask, tgt_nums, tgt_mask):
        return self.decoder(self.tgt_embed(tgt_nums), memory, src_mask, tgt_mask)