from layers import PositionalEncoding, MultiHeadAttention, \
    Embedding, LogSoftmax, ScaleNorm, FeedForward, clone
import torch, torch.nn as nn

class SublayerConnection(nn.Module):

    def __init__(self, scale, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(scale)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        scale = torch.tensor(embed_dim, dtype=torch.float32)
        self.sublayers = clone(SublayerConnection(torch.sqrt(scale), dropout), 2)

    def forward(self, x, src_mask):
        x = self.sublayers[0](x, lambda x: self.self_att(x, x, x, src_mask))
        return self.sublayers[1](x, self.ff)

class Encoder(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        scale = torch.tensor(embed_dim, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.crss_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        scale = torch.tensor(embed_dim, dtype=torch.float32)
        self.sublayers = clone(SublayerConnection(torch.sqrt(scale), dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_att(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.crss_att(x, m, m, src_mask))
        return self.sublayers[2](x, self.ff)

class Decoder(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        scale = torch.tensor(embed_dim, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))

    def forward(self, x, src_encs, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, src_encs, src_mask, tgt_mask)
        return self.norm(x)

class Model(nn.Module):

    def __init__(self, vocab_size, embed_dim=512, ff_dim=2048, num_heads=8, num_layers=6, dropout=0.3):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(embed_dim, ff_dim, num_heads, num_layers, dropout)
        self.decoder = Decoder(embed_dim, ff_dim, num_heads, num_layers, dropout)
        self.src_embed = nn.Sequential(Embedding(embed_dim, vocab_size), PositionalEncoding(embed_dim, dropout))
        self.tgt_embed = nn.Sequential(Embedding(embed_dim, vocab_size), PositionalEncoding(embed_dim, dropout))
        self.generator = LogSoftmax(embed_dim, vocab_size)

    def forward(self, src_nums, tgt_nums, src_mask, tgt_mask):
        return self.decode(self.encode(src_nums, src_mask), tgt_nums, src_mask, tgt_mask)

    def encode(self, src_nums, src_mask):
        return self.encoder(self.src_embed(src_nums), src_mask)

    def decode(self, src_encs, tgt_nums, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt_nums), src_encs, src_mask, tgt_mask)

def train_epoch(data, model, criterion, optimizer=None, *, mode='train'):
    total_loss = 0.
    for batch in data:
        src_nums, tgt_nums = batch.src_nums, batch.tgt_nums
        src_mask, tgt_mask = batch.src_mask, batch.tgt_mask

        logits = model(src_nums, tgt_nums[:, :-1], src_mask, tgt_mask)
        lprobs = torch.flatten(model.generator(logits), 0, 1)
        loss = criterion(lprobs, torch.flatten(tgt_nums[:, 1:]))

        if optimizer and mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() / batch.n_tokens
        del logits, lprobs, loss
    return total_loss
