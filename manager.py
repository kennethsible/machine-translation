import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            else self.word_to_num['<UNK>'] for word in words], device=device)
        return nums[0] if len(nums) == 1 else nums

    def denumberize(self, *nums):
        words = [self.num_to_word[num] for num in nums]
        return words[0] if len(words) == 1 else words

    def size(self):
        return len(self.num_to_word)

def subsequent_mask(size):
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0

class Batch:

    def __init__(self, src_nums, tgt_nums, padding_idx):
        self.src_nums = src_nums
        self.tgt_nums = tgt_nums
        self.src_mask = (self.src_nums != padding_idx).unsqueeze(-2)
        self.tgt_mask = (self.tgt_nums != padding_idx).unsqueeze(-2)
        self.n_tokens = self.tgt_mask.detach().sum()
        self.tgt_mask = self.tgt_mask & subsequent_mask(self.tgt_nums.size(-1))

def load_data(data_file, data_limit=None, batch_size=None, max_len=None):
        data = []
        with open(data_file) as file:
            for line in file:
                if data_limit and len(data) > data_limit: break
                src_line, tgt_line = line.split('\t')
                src_words = ['<BOS>'] + src_line.split() + ['<EOS>']
                tgt_words = ['<BOS>'] + tgt_line.split() + ['<EOS>']
                if not max_len or 2 < len(src_words) <= max_len and 2 < len(tgt_words) <= max_len:
                    data.append((src_words, tgt_words))
        data.sort(key=lambda x: len(x[0]))
        if batch_size is None: return data

        batched = []
        for i in range(batch_size, len(data) + 1, batch_size):
            batch = data[(i - batch_size):i]
            src_max_len = max(len(src_words) for src_words, _ in batch)
            tgt_max_len = max(len(tgt_words) for _, tgt_words in batch)
            for src_words, tgt_words in batch:
                src_res = src_max_len - len(src_words)
                tgt_res = tgt_max_len - len(tgt_words)
                if src_res > 0:
                    src_words.extend(src_res * ['<PAD>'])
                if tgt_res > 0:
                    tgt_words.extend(tgt_res * ['<PAD>'])
            batched.append(batch)
        return batched
