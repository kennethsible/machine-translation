from layers import Embedding, PositionalEncoding, \
    FeedForward, ScaleNorm, MultiHeadAttention, clone
import torch.nn as nn

class SublayerConnection(nn.Module):

    def __init__(self, embed_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(embed_dim ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 2)

    def forward(self, src_encs, src_mask):
        src_encs = self.sublayers[0](src_encs,
            lambda x: self.self_att(x, x, x, src_mask))
        return self.sublayers[1](src_encs, self.ff)

class Encoder(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        for p in self.parameters():
             if p.dim() > 1:
                 nn.init.xavier_uniform_(p)
        self.norm = ScaleNorm(embed_dim ** 0.5)

    def forward(self, src_embs, src_mask):
        src_encs = src_embs
        for layer in self.layers:
            src_encs = layer(src_encs, src_mask)
        return self.norm(src_encs)

class DecoderLayer(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.crss_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 3)

    def forward(self, tgt_encs, tgt_mask, src_encs, src_mask):
        tgt_encs = self.sublayers[0](tgt_encs,
            lambda x: self.self_att(x, x, x, tgt_mask))
        tgt_encs = self.sublayers[1](tgt_encs,
            lambda x: self.crss_att(x, src_encs, src_encs, src_mask))
        return self.sublayers[2](tgt_encs, self.ff)

class Decoder(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        for p in self.parameters():
             if p.dim() > 1:
                 nn.init.xavier_uniform_(p)
        self.norm = ScaleNorm(embed_dim ** 0.5)

    def forward(self, tgt_embs, tgt_mask, src_encs, src_mask):
        tgt_encs = tgt_embs
        for layer in self.layers:
            tgt_encs = layer(tgt_encs, tgt_mask, src_encs, src_mask)
        return self.norm(tgt_encs)

class Model(nn.Module):

    def __init__(self, vocab_dim, embed_dim, output_dim, ff_dim, num_heads, num_layers, dropout):
        super(Model, self).__init__()
        self.encoder = Encoder(embed_dim, ff_dim, num_heads, num_layers, dropout)
        self.decoder = Decoder(embed_dim, ff_dim, num_heads, num_layers, dropout)
        self.out_embed = Embedding(embed_dim, vocab_dim, output_dim)
        self.src_embed = nn.Sequential(self.out_embed, PositionalEncoding(embed_dim, dropout))
        self.tgt_embed = nn.Sequential(self.out_embed, PositionalEncoding(embed_dim, dropout))

    def encode(self, src_nums, src_mask):
        src_embs = self.src_embed(src_nums)
        return self.encoder(src_embs, src_mask)

    def decode(self, tgt_nums, tgt_mask, src_encs, src_mask):
        tgt_embs = self.tgt_embed(tgt_nums)
        return self.decoder(tgt_embs, tgt_mask, src_encs, src_mask)

    def forward(self, src_nums, src_mask, tgt_nums, tgt_mask):
        src_encs = self.encode(src_nums, src_mask)
        tgt_encs = self.decode(tgt_nums, tgt_mask, src_encs, src_mask)
        return self.out_embed(tgt_encs, inverse=True)
