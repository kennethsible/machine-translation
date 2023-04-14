from sacremoses import MosesTokenizer, MosesDetokenizer
from subword_nmt.apply_bpe import BPE
from model import Model
from decode import triu_mask
from io import StringIO
import torch, math, re
import torch.nn as nn

class Vocab:

    def __init__(self, vocab_file=None):
        self.num_to_word = []
        self.word_to_num = {}
        self.split_point = None
        if vocab_file:
            self._from_file(vocab_file)

    def _from_file(self, vocab_file):
        # <UNK> SRC-TGT <EOS> TGT-ONLY <BOS> SRC-ONLY <PAD>
        vocab_file.seek(0)
        line = vocab_file.readline().lstrip('#').split(';')
        tgt_loc, src_loc = int(line[0]), int(line[1])
        self.add('<UNK>')
        for i, line in enumerate(vocab_file):
            if i == src_loc - 2:
                self.add('<BOS>')
            elif i == tgt_loc - 2:
                self.add('<EOS>')
            self.add(line.split()[0])
        self.add('<PAD>')
        self.split = src_loc

    @property
    def UNK(self):
        try:
            return self.word_to_num['<UNK>']
        except ValueError:
            self.add('<UNK>')
        return self.word_to_num['<UNK>']

    @property
    def BOS(self):
        try:
            return self.word_to_num['<BOS>']
        except ValueError:
            self.add('<BOS>')
        return self.word_to_num['<BOS>']

    @property
    def EOS(self):
        try:
            return self.word_to_num['<EOS>']
        except ValueError:
            self.add('<EOS>')
        return self.word_to_num['<EOS>']

    @property
    def PAD(self):
        try:
            return self.word_to_num['<PAD>']
        except ValueError:
            self.add('<PAD>')
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
        if not self.ignore_index:
            return None
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

class Manager:

    def __init__(self, src_lang, tgt_lang, vocab_file, codes_file,
            model_file, config, device, data_file=None, test_file=None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._model_file = model_file
        self._vocab_file = vocab_file
        self._codes_file = codes_file
        self.config = config
        self.device = device

        for option, value in config.items():
            self.__setattr__(option, value)

        if not isinstance(vocab_file, str):
            vocab_file.seek(0)
            self._vocab_file = ''.join(vocab_file.readlines())
        self.vocab = Vocab(StringIO(self._vocab_file))
        if not isinstance(codes_file, str):
            codes_file.seek(0)
            self._codes_file = ''.join(codes_file.readlines())
        self.codes = BPE(StringIO(self._codes_file))

        self.model = Model(
            self.vocab.size(),
            self.embed_dim,
            self.ff_dim,
            self.num_heads,
            self.num_layers,
            self.dropout
        ).to(device)

        if data_file:
            data_file.seek(0)
            self.data = self.batch_data(data_file)
            assert len(self.data) > 0
        else:
            self.data = None

        if test_file:
            test_file.seek(0)
            self.test = self.batch_data(test_file)
            assert len(self.test) > 0
        else:
            self.test = None

    def tokenize(self, string, lang=None):
        if lang is None:
            lang = self.src_lang
        tokens = MosesTokenizer(lang).tokenize(string)
        return self.codes.process_line(' '.join(tokens))

    def detokenize(self, tokens, lang=None):
        if lang is None:
            lang = self.tgt_lang
        string = MosesDetokenizer(lang).detokenize(tokens)
        return re.sub('(@@ )|(@@ ?$)', '', string)

    def load_model(self):
        model_dict = torch.load(self._model_file, self.device)
        self.model.load_state_dict(model_dict['state_dict'])

    def save_model(self):
        model_dict = {
            'state_dict': self.model.state_dict(),
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang,
            'vocab_file': self._vocab_file,
            'codes_file': self._codes_file,
            'config': self.config
        }
        torch.save(model_dict, self._model_file)

    def batch_data(self, data_file):
        data_file.seek(0)

        unbatched, batched = [], []
        for line in data_file:
            src_line, tgt_line = line.split('\t')
            src_words = src_line.split()
            tgt_words = tgt_line.split()

            if not src_words or not tgt_words: continue
            if self.max_length:
                if len(src_words) > self.max_length - 2: continue
                if len(tgt_words) > self.max_length - 2: continue

            unbatched.append((
                ['<BOS>'] + src_words + ['<EOS>'],
                ['<BOS>'] + tgt_words + ['<EOS>']))
        unbatched.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        i = n = 0
        while i < len(unbatched):
            src_len = len(unbatched[i][0])
            tgt_len = len(unbatched[i][1])

            while True:
                n = self.batch_size // max(src_len, tgt_len)
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
