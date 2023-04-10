from layers import PositionalEncoding, MultiHeadAttention, \
    Embedding, LogSoftmax, FeedForward, ScaleNorm, clone
import torch, torch.nn as nn

class SublayerConnection(nn.Module):

    def __init__(self, embed_dim, dropout):
        super(SublayerConnection, self).__init__()
        scale = torch.tensor(embed_dim, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, sublayer):
        return inputs + self.dropout(sublayer(self.norm(inputs)))

class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 2)

    def forward(self, src_encs, src_mask):
        src_encs = self.sublayers[0](src_encs,
            lambda inputs: self.self_att(inputs, inputs, inputs, src_mask))
        return self.sublayers[1](src_encs, self.ff)

class Encoder(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        scale = torch.tensor(embed_dim, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))

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
            lambda inputs: self.self_att(inputs, inputs, inputs, tgt_mask))
        tgt_encs = self.sublayers[1](tgt_encs,
            lambda inputs: self.crss_att(inputs, src_encs, src_encs, src_mask))
        return self.sublayers[2](tgt_encs, self.ff)

class Decoder(nn.Module):

    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        scale = torch.tensor(embed_dim, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))

    def forward(self, tgt_embs, tgt_mask, src_encs, src_mask):
        tgt_encs = tgt_embs
        for layer in self.layers:
            tgt_encs = layer(tgt_encs, tgt_mask, src_encs, src_mask)
        return self.norm(tgt_encs)

class Model(nn.Module):

    def __init__(self, vocab_dim, embed_dim, ff_dim, num_heads, num_layers, dropout):
        super(Model, self).__init__()
        self.encoder = Encoder(embed_dim, ff_dim, num_heads, num_layers, dropout)
        self.decoder = Decoder(embed_dim, ff_dim, num_heads, num_layers, dropout)
        self.src_embed = nn.Sequential(Embedding(vocab_dim, embed_dim), PositionalEncoding(embed_dim, dropout))
        self.tgt_embed = nn.Sequential(Embedding(vocab_dim, embed_dim), PositionalEncoding(embed_dim, dropout))
        self.generator = LogSoftmax(embed_dim, vocab_dim)
        del self.src_embed[0].weights, self.tgt_embed[0].weights
        self.src_embed[0].weights = self.generator.weights
        self.tgt_embed[0].weights = self.generator.weights

    def encode(self, src_nums, src_mask):
        src_embs = self.src_embed(src_nums)
        return self.encoder(src_embs, src_mask)

    def decode(self, tgt_nums, tgt_mask, src_encs, src_mask):
        tgt_embs = self.tgt_embed(tgt_nums)
        return self.decoder(tgt_embs, tgt_mask, src_encs, src_mask)

    def forward(self, src_nums, src_mask, tgt_nums, tgt_mask, output_dim=None, log_softmax=False):
        src_encs = self.encode(src_nums, src_mask)
        tgt_encs = self.decode(tgt_nums, tgt_mask, src_encs, src_mask)
        return self.generator(tgt_encs, output_dim, log_softmax)
