import torch, torch.nn as nn, copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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

    def __init__(self, src_nums, tgt_nums, ignore_index=None):
        self._src_nums = src_nums
        self._tgt_nums = tgt_nums
        if ignore_index:
            self._src_mask = (src_nums != ignore_index).unsqueeze(-2)
            self._tgt_mask = (tgt_nums[:, :-1] != ignore_index).unsqueeze(-2) \
                & triu_mask(tgt_nums[:, :-1].size(-1))
            self._n_tokens = (tgt_nums[:, 1:] != ignore_index).sum()
        else:
            self._src_mask = None
            self._tgt_mask = triu_mask(tgt_nums[:, :-1].size(-1))
            self._n_tokens = tgt_nums[:, 1:].sum()

    @property
    def src_nums(self):
        return self._src_nums.to(device)

    @property
    def tgt_nums(self):
        return self._tgt_nums.to(device)

    @property
    def src_mask(self):
        return self._src_mask.to(device)

    @property
    def tgt_mask(self):
        return self._tgt_mask.to(device)

    @property
    def n_tokens(self):
        return self._n_tokens.item()

    def size(self):
        return self._src_nums.size(0)

def triu_mask(size, device=None):
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0

def load_data(data, vocab, max_length=None, batch_size=None):
    unbatched = []
    with open(data) as file:
        for line in file:
            src_line, tgt_line = line.split('\t')
            src_words = src_line.split()
            tgt_words = ['<BOS>'] + tgt_line.split() + ['<EOS>']

            if not src_words or not tgt_words: continue
            if max_length and len(src_words) > max_length:
                src_words = src_words[:max_length]
            if max_length and len(tgt_words) > max_length:
                tgt_words = tgt_words[:max_length]

            unbatched.append((src_words, tgt_words))
    unbatched.sort(key=lambda x: len(x[0]))

    batched = []
    if not batch_size: batch_size = 1
    for i in range(0, len(unbatched), batch_size):
        src_batch, tgt_batch = zip(*unbatched[i:(i + batch_size)])
        src_len = max(len(src_words) for src_words in src_batch)
        tgt_len = max(len(tgt_words) for tgt_words in tgt_batch)

        src_nums = torch.stack([
            nn.functional.pad(vocab.numberize(*src_words), (0, src_len - len(src_words)),
                value=vocab.pad) for src_words in src_batch
        ])
        tgt_nums = torch.stack([
            nn.functional.pad(vocab.numberize(*tgt_words), (0, tgt_len - len(tgt_words)),
                value=vocab.pad) for tgt_words in tgt_batch
        ])

        batched.append(Batch(src_nums, tgt_nums, vocab.pad))
    return batched
