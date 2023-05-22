from sacremoses import MosesTokenizer, MosesDetokenizer
from subword_nmt.apply_bpe import BPE
from io import TextIOWrapper, StringIO
from decoder import triu_mask
from model import Model
import torch.nn as nn
import torch, math, re

Tensor = torch.Tensor

class Vocab:

    def __init__(self, vocab_file: TextIOWrapper | list[str] | None = None):
        self.num_to_word = ['<UNK>', '<BOS>', '<EOS>', '<PAD>']
        self.word_to_num = {x: i for i, x in enumerate(self.num_to_word)}

        self.UNK = self.word_to_num['<UNK>']
        self.BOS = self.word_to_num['<BOS>']
        self.EOS = self.word_to_num['<EOS>']
        self.PAD = self.word_to_num['<PAD>']

        if vocab_file:
            for line in vocab_file:
                self.add(line.split()[0])

    def add(self, word: str):
        if word not in self.word_to_num:
            self.word_to_num[word] = self.size()
            self.num_to_word.append(word)

    def numberize(self, words: list[str]) -> list[int]:
        return [self.word_to_num[word] if word in self.word_to_num
            else self.UNK for word in words]

    def denumberize(self, nums: list[int]) -> list[str]:
        return [self.num_to_word[num] for num in nums]

    def size(self) -> int:
        return len(self.num_to_word)

class Batch:

    def __init__(self, src_nums: Tensor, tgt_nums: Tensor, ignore_index: int, device: str):
        self._src_nums = src_nums
        self._tgt_nums = tgt_nums
        self.ignore_index = ignore_index
        self.device = device

    @property
    def src_nums(self) -> Tensor:
        return self._src_nums.to(self.device)

    @property
    def tgt_nums(self) -> Tensor:
        return self._tgt_nums.to(self.device)

    @property
    def src_mask(self) -> Tensor:
        return (self.src_nums != self.ignore_index).unsqueeze(-2)

    @property
    def tgt_mask(self) -> Tensor:
        return triu_mask(self.tgt_nums[:, :-1].size(-1), device=self.device) \
            & (self.tgt_nums[:, :-1] != self.ignore_index).unsqueeze(-2)

    def length(self) -> int:
        return int((self.tgt_nums[:, 1:] != self.ignore_index).sum())

    def size(self) -> int:
        return self._src_nums.size(0)

class Manager:

    embed_dim: int
    ff_dim: int
    num_heads: int
    dropout: float
    num_layers: int
    max_epochs: int
    lr: float
    patience: int
    decay_factor: float
    min_lr: float
    label_smoothing: float
    clip_grad: float
    batch_size: int
    max_length: int
    beam_size: int

    def __init__(self, src_lang: str, tgt_lang: str, vocab_file: TextIOWrapper | list[str], codes_file: TextIOWrapper | list[str],
            model_file: str, config: dict, device: str, data_file: TextIOWrapper | None = None, test_file: TextIOWrapper | None = None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._model_file = model_file
        self._vocab_file = vocab_file
        self._codes_file = codes_file
        self.config = config
        self.device = device

        for option, value in config.items():
            self.__setattr__(option, value)

        if isinstance(vocab_file, TextIOWrapper):
            self._vocab_file = list(vocab_file.readlines())
        self.vocab = Vocab(self._vocab_file)

        if isinstance(codes_file, TextIOWrapper):
            self._codes_file = list(codes_file.readlines())
        self.codes = BPE(StringIO(''.join(self._codes_file)))

        self.model = Model(
            self.vocab.size(),
            self.embed_dim,
            self.ff_dim,
            self.num_heads,
            self.dropout,
            self.num_layers
        ).to(device)

        self.data = None
        if data_file:
            self.data = self.batch_data(data_file)

        self.test = None
        if test_file:
            self.test = self.batch_data(test_file)

    def tokenize(self, text: str, lang: str | None = None) -> str:
        if lang is None:
            lang = self.src_lang
        tokens = MosesTokenizer(lang).tokenize(text)
        return self.codes.process_line(' '.join(tokens))

    def detokenize(self, tokens: list[str], lang: str | None = None) -> str:
        if lang is None:
            lang = self.tgt_lang
        string = MosesDetokenizer(lang).detokenize(tokens)
        return re.sub('(@@ )|(@@ ?$)', '', string)

    def save_model(self):
        torch.save({
            'state_dict': self.model.state_dict(),
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang,
            'vocab_file': self._vocab_file,
            'codes_file': self._codes_file,
            'config': self.config
        }, self._model_file)

    def batch_data(self, data_file: TextIOWrapper) -> list[Batch]:
        unbatched, batched = [], []
        for line in data_file:
            src_line, tgt_line = line.split('\t')
            src_words = src_line.split()
            tgt_words = tgt_line.split()

            if not src_words or not tgt_words: continue
            if self.max_length:
                if len(src_words) > self.max_length - 2:
                    src_words = src_words[:self.max_length]
                if len(tgt_words) > self.max_length - 2:
                    tgt_words = tgt_words[:self.max_length]

            unbatched.append((
                ['<BOS>'] + src_words + ['<EOS>'],
                ['<BOS>'] + tgt_words + ['<EOS>']))

        unbatched.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        i = batch_size = 0
        while (i := i + batch_size) < len(unbatched):
            src_len = len(unbatched[i][0])
            tgt_len = len(unbatched[i][1])

            while True:
                batch_size = self.batch_size // max(src_len, tgt_len)
                batch_size = 2 ** math.floor(math.log2(batch_size))

                src_batch, tgt_batch = zip(*unbatched[i:(i + batch_size)])
                max_src_len = max(len(src_words) for src_words in src_batch)
                max_tgt_len = max(len(tgt_words) for tgt_words in tgt_batch)

                if src_len >= max_src_len and tgt_len >= max_tgt_len: break
                src_len, tgt_len = max_src_len, max_tgt_len

            max_src_len = math.ceil(max_src_len / 8) * 8
            max_tgt_len = math.ceil(max_tgt_len / 8) * 8

            src_nums = torch.stack([nn.functional.pad(torch.tensor(self.vocab.numberize(src_words)),
                (0, max_src_len - len(src_words)), value=self.vocab.PAD) for src_words in src_batch])
            tgt_nums = torch.stack([nn.functional.pad(torch.tensor(self.vocab.numberize(tgt_words)),
                (0, max_tgt_len - len(tgt_words)), value=self.vocab.PAD) for tgt_words in tgt_batch])
            batched.append(Batch(src_nums, tgt_nums, self.vocab.PAD, self.device))

        return batched
