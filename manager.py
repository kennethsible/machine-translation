from sacremoses import MosesTokenizer, MosesDetokenizer
from subword_nmt.apply_bpe import BPE
from model import Model
import torch, torch.nn as nn
import math, re

class Tokenizer:

    def __init__(self, src_lang, tgt_lang=None, codes_file=None):
        self.mt = MosesTokenizer(src_lang)
        if tgt_lang:
            self.md = MosesDetokenizer(tgt_lang)
        else:
            self.md = MosesDetokenizer(src_lang)
        if codes_file:
            with open(codes_file) as file:
                self.bpe = BPE(file)
        else:
            self.bpe = None

    def tokenize(self, text):
        string = self.mt.tokenize(text, return_str=True)
        return self.bpe.process_line(string)

    def detokenize(self, tokens):
        string = self.md.detokenize(tokens)
        return re.sub('(@@ )|(@@ ?$)', '', string)

class Vocab:

    def __init__(self, initialize=True):
        if initialize:
            self.num_to_word = ['<UNK>', '<BOS>', '<EOS>', '<PAD>']
            self.word_to_num = {word: i for i, word in enumerate(self.num_to_word)}
        else:
            self.num_to_word = []
            self.word_to_num = {}

    @property
    def UNK(self):
        return self.word_to_num['<UNK>']

    @property
    def BOS(self):
        return self.word_to_num['<BOS>']

    @property
    def EOS(self):
        return self.word_to_num['<EOS>']

    @property
    def PAD(self):
        return self.word_to_num['<PAD>']

    def add(self, word):
        try:
            self.word_to_num[word] = self.size()
        except ValueError:
            pass
        else:
            self.num_to_word.append(word)      

    def remove(self, word):
        try:
            self.word_to_num.pop(word)
        except KeyError:
            pass
        else:
            self.num_to_word.remove(word)

    def numberize(self, *words, as_list=False):
        nums = []
        for word in words:
            try:
                nums.append(self.word_to_num[word])
            except KeyError:
                nums.append(self.UNK)
        return nums if as_list else torch.tensor(nums)

    def denumberize(self, *nums, verbatim=False):
        if verbatim:
            return [self.num_to_word[num] for num in nums]
        try:
            start = nums.index(self.BOS) + 1
        except ValueError:
            start = 0
        try:
            end = nums.index(self.EOS)
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
    def num_tokens(self):
        if not self.ignore_index:
            return self.tgt_nums[:, 1:].sum()
        return (self.tgt_nums[:, 1:] != self.ignore_index).sum()

    def size(self):
        return self._src_nums.size(0)

def triu_mask(size, device=None):
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0

class Manager:

    def __init__(self, src_lang, tgt_lang, config, device, vocab_file, data_file=None, test_file=None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.config = config
        self.device = device

        self.vocab = Vocab(initialize=False)
        with open(vocab_file) as file:
            # <UNK> SRC-TGT <EOS> TGT-ONLY <BOS> SRC-ONLY <PAD>
            tgt_loc, src_loc = file.readline().lstrip('#').split(';')
            tgt_loc, src_loc = int(tgt_loc), int(src_loc)
            self.vocab.add('<UNK>')
            for i, line in enumerate(file):
                if i == src_loc - 2:
                    self.vocab.add('<BOS>')
                elif i == tgt_loc - 2:
                    self.vocab.add('<EOS>')
                self.vocab.add(line.split()[0])
            self.vocab.add('<PAD>')
        self.output_dim = src_loc

        self.model = Model(
            self.vocab.size(),
            config['embed_dim'],
            config['ff_dim'],
            config['num_heads'],
            config['num_layers'],
            config['dropout']
        ).to(device)

        if data_file:
            self.data = self.load_data(data_file)
            assert len(self.data) > 0
        else:
            self.data = None

        if test_file:
            self.test = self.load_data(test_file)
            assert len(self.test) > 0
        else:
            self.test = None

    def load_model(self, model_file):
        state_dict = torch.load(model_file, self.device)
        self.model.load_state_dict(state_dict)

    def save_model(self, model_file):
        state_dict = self.model.state_dict()
        torch.save(state_dict, model_file)

    def load_data(self, data):
        max_length = self.config['max_length']
        batch_size = self.config['batch_size']

        unbatched, batched = [], []
        with open(data) as file:
            for line in file:
                src_line, tgt_line = line.split('\t')
                src_words = src_line.split()
                tgt_words = tgt_line.split()

                if not src_words or not tgt_words: continue
                if max_length:
                    if len(src_words) > max_length - 2: continue
                    if len(tgt_words) > max_length - 2: continue

                unbatched.append((
                    ['<BOS>'] + src_words + ['<EOS>'],
                    ['<BOS>'] + tgt_words + ['<EOS>']))
        unbatched.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        i = n = 0
        while i < len(unbatched):
            src_len = len(unbatched[i][0])
            tgt_len = len(unbatched[i][1])

            while True:
                n = batch_size // max(src_len, tgt_len)
                if n & (n - 1) != 0:
                    n = 2 ** (math.ceil(math.log2(n)) - 1)
    
                src_batch, tgt_batch = zip(*unbatched[i:(i + n)])
                max_src_len = max(len(src_words) for src_words in src_batch)
                max_tgt_len = max(len(tgt_words) for tgt_words in tgt_batch)

                if (src_len >= max_src_len and tgt_len >= max_tgt_len): break
                src_len, tgt_len = max_src_len, max_tgt_len

            src_nums = torch.stack([
                nn.functional.pad(self.vocab.numberize(*src_words), (0, max_src_len - len(src_words)),
                    value=self.vocab.PAD) for src_words in src_batch])
            tgt_nums = torch.stack([
                nn.functional.pad(self.vocab.numberize(*tgt_words), (0, max_tgt_len - len(tgt_words)),
                    value=self.vocab.PAD) for tgt_words in tgt_batch])

            batched.append(Batch(src_nums, tgt_nums, self.device, self.vocab.PAD))
            i += n
        return batched
