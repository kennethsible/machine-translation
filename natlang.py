import torch

class Vocab:

    def __init__(self):
        self.num_to_word = ['<PAD>', '<EOS>', '<UNK>']
        self.word_to_num = {word: i for i, word in enumerate(self.num_to_word)}

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
        nums = [self.word_to_num[word] if word in self.word_to_num
            else self.word_to_num['<UNK>'] for word in words]
        return torch.tensor(nums) if len(nums) > 1 else torch.tensor(nums[:1])

    def denumberize(self, *nums):
        words = [self.num_to_word[num] for num in nums]
        return words if len(words) > 1 else words[0]

    def __len__(self):
        return len(self.num_to_word)

def bmv(w, x):
    x = x.unsqueeze(-1)
    y = w @ x
    y = y.squeeze(-1)
    return y

class Embedding(torch.nn.Module):

    def __init__(self, vocab_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(vocab_size, output_size))
        torch.nn.init.normal_(self.W, std=0.01)

    def forward(self, input):
        emb = self.W[input.transpose(0, 1)]
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
        return bmv(self.W, input) + self.b

class LayerNorm(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_size, input_size))
        self.b = torch.nn.Parameter(torch.empty(output_size))
        torch.nn.init.normal_(self.W, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def forward(self, input):
        input = (input - torch.mean(input)) / torch.std(input)
        return bmv(self.W, input) + self.b

class Tanh(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_size, input_size))
        self.b = torch.nn.Parameter(torch.empty(output_size))
        torch.nn.init.normal_(self.W, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def forward(self, input):
        z = bmv(self.W, input) + self.b
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
        z = bmv(W, input)
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
        z = bmv(self.W_hi, input) + bmv(self.W_hh, hidden) + self.b
        return torch.tanh(z)

class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.layers = torch.nn.ModuleList([RNNCell(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        for rnn in self.layers:
            h, H = rnn.h0.repeat(batch_size, 1), []
            for input in inputs:
                h = rnn(input, h)
                H.append(h)
            inputs = torch.stack(H)
        return inputs

class LSTMCell(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.h0 = torch.nn.Parameter(torch.empty(hidden_size))
        self.m0 = torch.nn.Parameter(torch.empty(hidden_size))
        self.W_xf = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hf = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xi = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hi = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xo = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_ho = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xm = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hm = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_f = torch.nn.Parameter(torch.empty(hidden_size))
        self.b_i = torch.nn.Parameter(torch.empty(hidden_size))
        self.b_o = torch.nn.Parameter(torch.empty(hidden_size))
        self.b_m = torch.nn.Parameter(torch.empty(hidden_size))
        torch.nn.init.normal_(self.h0, std=0.01)
        torch.nn.init.normal_(self.m0, std=0.01)
        torch.nn.init.normal_(self.W_xf, std=0.01)
        torch.nn.init.normal_(self.W_hf, std=0.01)
        torch.nn.init.normal_(self.W_xi, std=0.01)
        torch.nn.init.normal_(self.W_hi, std=0.01)
        torch.nn.init.normal_(self.W_xo, std=0.01)
        torch.nn.init.normal_(self.W_ho, std=0.01)
        torch.nn.init.normal_(self.W_xm, std=0.01)
        torch.nn.init.normal_(self.W_hm, std=0.01)
        torch.nn.init.normal_(self.b_f, std=0.01)
        torch.nn.init.normal_(self.b_i, std=0.01)
        torch.nn.init.normal_(self.b_o, std=0.01)
        torch.nn.init.normal_(self.b_m, std=0.01)

    def forward(self, input, hidden, memory):
        f = torch.sigmoid(bmv(self.W_xf, input) + bmv(self.W_hf, hidden) + self.b_f)
        i = torch.sigmoid(bmv(self.W_xi, input) + bmv(self.W_hi, hidden) + self.b_i)
        o = torch.sigmoid(bmv(self.W_xo, input) + bmv(self.W_ho, hidden) + self.b_o)
        m = f * memory + i * torch.tanh(bmv(self.W_xm, input) + bmv(self.W_hm, hidden) + self.b_m)
        return o * torch.tanh(m), m

class LSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.layers = torch.nn.ModuleList([LSTMCell(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        for lstm in self.layers:
            h, m, H = lstm.h0.repeat(batch_size, 1), lstm.m0.repeat(batch_size, 1), []
            for input in inputs:
                h, m = lstm(input, h, m)
                H.append(h)
            inputs = torch.stack(H)
        return inputs

def attention(query, keys, values, mask):
    # (torch.zeros((10, 256)).unsqueeze(1) @ torch.zeros((10, 256, 18))).squeeze(1).size()
    scores  = query.unsqueeze(1) @ keys.transpose(-2, -1)
    # print(scores.squeeze(1) * mask) TODO
    weights = torch.softmax(scores.squeeze(1), dim=-1)
    context = weights.unsqueeze(1) @ values
    return context.squeeze(1)

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
