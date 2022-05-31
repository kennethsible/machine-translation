import torch
from collections.abc import MutableSet

class Vocab(MutableSet):

    def __init__(self):
        super().__init__()
        self.num_to_word = ['<SEP>', '<EOS>', '<UNK>']
        self.word_to_num = {word: i for i, word in enumerate(self.num_to_word)}

    def add(self, word):
        if word not in self.word_to_num:
            num = len(self.num_to_word)
            self.num_to_word.append(word)
            self.word_to_num[word] = num

    def discard(self, word):
        if word in self.word_to_num:
            self.num_to_word.remove(word)
            self.word_to_num.pop(word)

    def numberize(self, *words):
        nums = [self.word_to_num[word] if word in self.word_to_num
            else self.word_to_num['<UNK>'] for word in words]
        return torch.tensor(nums) if len(nums) > 1 else torch.tensor(nums[:1])

    def denumberize(self, *nums):
        words = [self.num_to_word[num] for num in nums]
        return words if len(words) > 1 else words[0]

    def __contains__(self, word):
        return word in self.word_to_num

    def __iter__(self):
        return iter(self.num_to_word)

    def __len__(self):
        return len(self.num_to_word)

class Embedding(torch.nn.Module):

    def __init__(self, vocab_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(vocab_size, output_size))
        torch.nn.init.normal_(self.W, std=0.01)

    def forward(self, input):
        emb = self.W[input]
        # https://www.aclweb.org/anthology/N18-1031/
        return torch.nn.functional.normalize(emb, dim=1)

class Tanh(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_size, input_size))
        self.b = torch.nn.Parameter(torch.empty(output_size))
        torch.nn.init.normal_(self.W, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def forward(self, input):
        z = self.W @ input + self.b
        return torch.tanh(z)

class LogSoftmax(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_size, input_size))
        torch.nn.init.normal_(self.W, std=0.01)

    def forward(self, input):
        # https://www.aclweb.org/anthology/N18-1031/
        W = torch.nn.functional.normalize(self.W, dim=-1)
        input = 10 * torch.nn.functional.normalize(input, dim=-1)
        z = W @ input
        return torch.log_softmax(z, dim=-1)

class RNNCell(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.h0 = torch.nn.Parameter(torch.empty(hidden_size))
        self.W_hi = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_hc = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = torch.nn.Parameter(torch.empty(hidden_size))
        torch.nn.init.normal_(self.h0, std=0.01)
        torch.nn.init.normal_(self.W_hi, std=0.01)
        torch.nn.init.normal_(self.W_hh, std=0.01)
        torch.nn.init.normal_(self.W_hc, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def forward(self, input, hidden):
        return torch.tanh(self.W_hi @ input + self.W_hh @ hidden + self.b)

class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, stack=1):
        super().__init__()
        self.rnn = torch.nn.ModuleList([RNNCell(input_size, hidden_size) for _ in range(stack)])

    def forward(self, input):
        for i in range(len(self.rnn)):
            H = [self.rnn[i].h0]
            for emb in input:
                H.append(self.rnn[i](emb, H[-1]))
            input = torch.stack(H)
        return torch.stack(H)

def attention(query, keys, values):
    scores  = query @ keys.transpose(-2, -1)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ values
    return context
