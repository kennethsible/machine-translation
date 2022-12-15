from sacremoses import MosesTokenizer, MosesDetokenizer
from subword_nmt.apply_bpe import BPE
from layers import CrossEntropy
from model import Model, train_epoch
import torch, torch.nn as nn
import math, re

class Tokenizer:

    def __init__(self, src_lang, tgt_lang, codes_file=None):
        self.mt = MosesTokenizer(src_lang)
        self.md = MosesDetokenizer(tgt_lang)

        if codes_file:
            with open(codes_file) as file:
                self.bpe = BPE(file)
        else: self.bpe = None

    def tokenize(self, input):
        string = self.mt.tokenize(input, return_str=True)
        return self.bpe.process_line(string)

    def detokenize(self, output):
        string = self.md.detokenize(output)
        return re.sub('(@@ )|(@@ ?$)', '', string)

class Vocab:

    def __init__(self):
        self.num_to_word = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
        self.word_to_num = {word: i for i, word in enumerate(self.num_to_word)}
        self.bos = self.word_to_num['<BOS>']
        self.eos = self.word_to_num['<EOS>']
        self.pad = self.word_to_num['<PAD>']
        self.unk = self.word_to_num['<UNK>']

    def add(self, word):
        if word not in self.word_to_num:
            num = len(self.num_to_word)
            self.num_to_word.append(word)
            self.word_to_num[word] = num

    def remove(self, word):
        if word in self.word_to_num:
            self.num_to_word.remove(word)
            self.word_to_num.pop(word)

    def numberize(self, *words, as_tensor=True):
        nums = [self.word_to_num[word] if word in self.word_to_num
            else self.unk for word in words]
        return torch.tensor(nums) if as_tensor else nums

    def denumberize(self, *nums, strip=True):
        if not strip:
            return [self.num_to_word[num] for num in nums]
        try:
            start = nums.index(self.bos) + 1
        except ValueError:
            start = 0
        try:
            end = nums.index(self.eos)
        except ValueError:
            end = len(nums)
        return [self.num_to_word[num] for num in nums[start:end]]

    def size(self):
        return len(self.num_to_word)

class Batch:

    def __init__(self, src_nums, tgt_nums, device=None, ignore_index=None):
        self._src_nums = src_nums
        self._tgt_nums = tgt_nums
        self.device = device
        self.ignore_index = ignore_index

    @property
    def src_nums(self):
        return self._src_nums.to(self.device)

    @property
    def tgt_nums(self):
        return self._tgt_nums.to(self.device)

    @property
    def src_mask(self):
        if not self.ignore_index: return None
        return (self.src_nums != self.ignore_index).unsqueeze(-2)

    @property
    def tgt_mask(self):
        if not self.ignore_index:
            return triu_mask(self.tgt_nums[:, :-1].size(-1))
        return (self.tgt_nums[:, :-1] != self.ignore_index).unsqueeze(-2) \
            & triu_mask(self.tgt_nums[:, :-1].size(-1), device=self.device)

    @property
    def n_tokens(self):
        if not self.ignore_index:
            return self.tgt_nums[:, 1:].sum()
        return (self.tgt_nums[:, 1:] != self.ignore_index).sum()

    def size(self):
        return self._src_nums.size(0)

def triu_mask(size, device=None):
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0

class Manager:

    def __init__(self, src_lang, tgt_lang, config=None, device=None,
            data_file=None, test_file=None, vocab_file=None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.config = config
        self.device = device

        if vocab_file:
            self.vocab = Vocab()
            with open(vocab_file) as file:
                for line in file:
                    self.vocab.add(line.split()[0])
            assert self.vocab.size() > 0
        else: self.vocab = None

        self.model = Model(self.vocab.size()).to(device)

        if data_file:
            self.data = self.load_data(data_file)
            assert len(self.data) > 0
        else: self.data = None
        if test_file:
            self.test = self.load_data(test_file)
            assert len(self.test) > 0
        else: self.test = None

    def load_model(self, model_file):
        state_dict = torch.load(model_file, self.device)
        self.model.load_state_dict(state_dict)

    def save_model(self, model_file):
        state_dict = self.model.state_dict()
        torch.save(state_dict, model_file)

    def batch_size_search(self):
        if not next(self.model.parameters()).is_cuda:
            raise RuntimeError('cannot optimize batch size on CPU (only CUDA)')

        criterion = CrossEntropy(self.config['smoothing'])
        optimizer = torch.optim.Adam(self.model.parameters(), self.config['lr'])
        self.model.train()

        batch_size, max_length = 1, self.config['max-length']
        while True:
            src_nums = torch.zeros((batch_size * 2, max_length), dtype=torch.long)
            tgt_nums = torch.zeros((batch_size * 2, max_length), dtype=torch.long)
            batch = Batch(src_nums, tgt_nums, self.device, self.vocab.pad)
            try:
                train_epoch(
                    [batch],
                    self.model,
                    criterion,
                    optimizer,
                    mode='train'
                )
            except RuntimeError:
                return batch_size
            batch_size *= 2

    def load_data(self, data):
        max_length = self.config['max-length']
        batch_size = self.config['batch-size']
        mem_fill = batch_size == -1
        if mem_fill:
            mem_limit = max_length * self.batch_size_search()
        max_length -= 2

        unbatched = []
        with open(data) as file:
            for line in file:
                src_line, tgt_line = line.split('\t')
                src_words = src_line.split()
                tgt_words = tgt_line.split()

                if not src_words or not tgt_words: continue
                if max_length and len(src_words) > max_length:
                    src_words = src_words[:max_length]
                if max_length and len(tgt_words) > max_length:
                    tgt_words = tgt_words[:max_length]

                unbatched.append((
                    ['<BOS>'] + src_words + ['<EOS>'],
                    ['<BOS>'] + tgt_words + ['<EOS>']
                ))
        unbatched.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        i, batched = 0, []
        while i < len(unbatched):
            src_len = len(unbatched[i][0])
            tgt_len = len(unbatched[i][1])
            while True:
                if mem_fill:
                    batch_size = mem_limit // max(src_len, tgt_len)
                    if batch_size & (batch_size - 1) != 0:
                        batch_size = 2 ** (math.ceil(math.log2(batch_size)) - 1)
                src_batch, tgt_batch = zip(*unbatched[i:(i + batch_size)])
                max_src_len = max(len(src_words) for src_words in src_batch)
                max_tgt_len = max(len(tgt_words) for tgt_words in tgt_batch)
                if mem_fill:
                    if (src_len >= max_src_len and tgt_len >= max_tgt_len): break # TODO
                    src_len, tgt_len = max_src_len, max_tgt_len
                else: break

            src_nums = torch.stack([
                nn.functional.pad(self.vocab.numberize(*src_words), (0, max_src_len - len(src_words)),
                    value=self.vocab.pad) for src_words in src_batch
            ])
            tgt_nums = torch.stack([
                nn.functional.pad(self.vocab.numberize(*tgt_words), (0, max_tgt_len - len(tgt_words)),
                    value=self.vocab.pad) for tgt_words in tgt_batch
            ])

            batched.append(Batch(src_nums, tgt_nums, self.device, self.vocab.pad))
            i += batch_size
        return batched
