import copy, torch, torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Vocab:

    def __init__(self):
        self.num_to_word = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
        self.word_to_num = {word: i for i, word in enumerate(self.num_to_word)}
        self.padding_idx = self.word_to_num['<PAD>']
        self.default_idx = self.word_to_num['<UNK>']

    def add(self, word):
        if word not in self.word_to_num:
            num = len(self.num_to_word)
            self.num_to_word.append(word)
            self.word_to_num[word] = num

    def remove(self, word):
        if word in self.word_to_num:
            self.num_to_word.remove(word)
            self.word_to_num.pop(word)

    def numberize(self, *words):
        nums = torch.tensor([self.word_to_num[word] if word in self.word_to_num
            else self.word_to_num['<UNK>'] for word in words])
        return nums[0] if len(nums) == 1 else nums

    def denumberize(self, *nums):
        words = [self.num_to_word[num] for num in nums]
        return words[0] if len(words) == 1 else words

    def size(self):
        return len(self.num_to_word)

class Batch:

    def __init__(self, src_nums, tgt_nums, padding_idx):
        self._src_nums = src_nums if src_nums.dim() > 1 else src_nums.unsqueeze(0)
        self._tgt_nums = tgt_nums if tgt_nums.dim() > 1 else tgt_nums.unsqueeze(0)

        self._tgt_mask = (self._tgt_nums[:, :-1] != padding_idx).unsqueeze(-2)
        self._n_tokens = self._tgt_mask.sum()

        self._src_mask = (self._src_nums != padding_idx).unsqueeze(-2)
        self._tgt_mask = self._tgt_mask & triu_mask(self._tgt_nums[:, :-1].size(-1), self._tgt_mask.device)

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

def load_data(data_file, vocab, batch_size=None, max_len=None):
    unbatched = []
    with open(data_file) as file:
        for line in file:
            src_line, tgt_line = line.split('\t')
            src_words, tgt_words = src_line.split(), tgt_line.split()

            if len(src_words) <= 1 or len(tgt_words) <= 1: continue
            if max_len and len(src_words) > max_len:
                src_words = src_words[:max_len]
            if max_len and len(tgt_words) > max_len:
                tgt_words = tgt_words[:max_len]

            for words in (src_words, tgt_words):
                words[:] = ['<BOS>'] + words + ['<EOS>']
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
                value=vocab.padding_idx) for src_words in src_batch
        ])
        tgt_nums = torch.stack([
            nn.functional.pad(vocab.numberize(*tgt_words), (0, tgt_len - len(tgt_words)),
                value=vocab.padding_idx) for tgt_words in src_batch
        ])

        batched.append(Batch(src_nums, tgt_nums, vocab.padding_idx))
    return batched
