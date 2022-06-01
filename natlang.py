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
        return torch.nn.functional.normalize(emb, dim=-1)

class Linear(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_size, input_size))
        self.b = torch.nn.Parameter(torch.empty(output_size))
        torch.nn.init.normal_(self.W, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def forward(self, input):
        return self.W @ input + self.b

class LayerNorm(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_size, input_size))
        self.b = torch.nn.Parameter(torch.empty(output_size))
        torch.nn.init.normal_(self.W, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def forward(self, input):
        input = (input - torch.mean(input)) / torch.std(input)
        return self.W @ input + self.b

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
        z = self.W_hi @ input + self.W_hh @ hidden + self.b
        return torch.tanh(z)

class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.layers = torch.nn.ModuleList([RNNCell(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, inputs):
        for rnn in self.layers:
            H = [rnn.h0]
            for input in inputs:
                H.append(rnn(input, H[-1]))
            inputs = torch.stack(H)
        return inputs

def attention(query, keys, values):
    scores  = query @ keys.transpose(-2, -1)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ values
    return context

class SelfAttention(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.W_Q = torch.nn.Parameter(torch.empty(output_size, input_size))
        self.W_K = torch.nn.Parameter(torch.empty(output_size, input_size))
        self.W_V = torch.nn.Parameter(torch.empty(output_size, input_size))
        torch.nn.init.normal_(self.W_Q, std=0.01)
        torch.nn.init.normal_(self.W_K, std=0.01)
        torch.nn.init.normal_(self.W_V, std=0.01)

    def forward(self, input):
        q = self.W_Q @ input
        k = self.W_K @ input
        v = self.W_V @ input
        return attention(q, k, v)
