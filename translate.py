import torch, random, math, copy, tqdm
from torchtext.data.metrics import bleu_score
from collections.abc import MutableSet
from collections import Counter

device = torch.device('cuda')

class Vocab(MutableSet):

    def __init__(self):
        super().__init__()

        self.num_to_word = ['<SEP', '<EOS>', '<UNK>']
        self.word_to_num = {word: index for index, word in enumerate(self.num_to_word)}

    def add(self, word):
        if word in self.word_to_num: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num

    def discard(self, word):
        if word in self.word_to_num:
            self.num_to_word.remove(word)
            self.word_to_num.pop(word)

    def numberize(self, word):
        if word in self.word_to_num:
            return self.word_to_num[word]
        return self.word_to_num['<UNK>']

    def denumberize(self, num):
        return self.num_to_word[num]

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
        z = self.W @ input
        return torch.log_softmax(z, dim=-1)

class Linear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_size, input_size))
        self.b = torch.nn.Parameter(torch.empty(output_size))
        torch.nn.init.normal_(self.W, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)
        self.bias = bias

    def forward(self, input):
        v = input.squeeze(0)
        z = self.W @ v
        if self.bias: z += self.b
        return z.unsqueeze(0)

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

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = RNNCell(input_size, hidden_size)

    def forward(self, input):
        h_seq = [self.rnn.h0]
        for emb in input:
            h_seq.append(self.rnn(emb, h_seq[-1]))
        return torch.stack(h_seq)

def attention(h, h_seq):
    scores  = h @ h_seq.transpose(-2, -1)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ h_seq
    return context

class Encoder(torch.nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.emb = Embedding(vocab_size, hidden_size)
        self.rnn = RNN(hidden_size, hidden_size)

    def forward(self, src_nums):
        emb = self.emb(src_nums)
        h_seq = self.rnn(emb)
        return h_seq

class Decoder(torch.nn.Module):
    
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.emb = Embedding(vocab_size, hidden_size)
        self.rnn = RNNCell(hidden_size, hidden_size)
        self.tnh = Tanh(2*hidden_size, hidden_size)
        self.out = LogSoftmax(hidden_size, vocab_size)

    def start(self, h_seq):
        h = self.rnn.h0
        return (h, h_seq) # (h, h_seq[-1])
        # return (c, c.clone().detach())

    def input(self, state, tgt_num):
        h, h_seq = state
        emb = self.emb(tgt_num).squeeze(0)
        h = self.rnn(emb, h)
        return (h, h_seq)

    def output(self, state):
        h, h_seq = state
        c = attention(h, h_seq)
        h = torch.cat((c, h), dim=-1)
        z = self.tnh(h)
        y = self.out(z)
        return y

class Model(torch.nn.Module):

    def __init__(self, src_vocab, tgt_vocab, hidden_size):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder = Encoder(len(src_vocab), hidden_size)
        self.decoder = Decoder(hidden_size, len(tgt_vocab))

    def translate(self, src_words):
        src_nums = torch.tensor([self.src_vocab.numberize(word)
            for word in src_words], device=device)
        src_encs = self.encoder(src_nums)
        state = self.decoder.start(src_encs)
        tgt_words = []
        for _ in range(100):
            out = self.decoder.output(state)
            tgt_num = torch.argmax(out).item()
            tgt_word = self.tgt_vocab.denumberize(tgt_num)
            if tgt_word == '<EOS>': break
            tgt_words.append(tgt_word)
            tgt_num = torch.tensor([tgt_num], device=device)
            state = self.decoder.input(state, tgt_num)
        return tgt_words

    def forward(self, src_words, tgt_words):
        src_nums = torch.tensor([self.src_vocab.numberize(word)
            for word in src_words], device=device)
        src_encs = self.encoder(src_nums)
        state = self.decoder.start(src_encs)
        loss = 0
        for tgt_word in tgt_words:
            out = self.decoder.output(state)
            tgt_num = torch.tensor([self.tgt_vocab.numberize(tgt_word)], device=device)
            loss -= out[tgt_num]
            state = self.decoder.input(state, tgt_num)
        loss /= len(tgt_words)
        return loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('--dev', type=str, help='development data')
    parser.add_argument('infile', nargs='?', type=str, help='test data to translate')
    parser.add_argument('-o', '--outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
    args = parser.parse_args()

    if args.train:
        train_data = []
        for line in open(args.train):
            lsplit = line.split('\t')
            src_line, tgt_line = lsplit[0].lower(), lsplit[1].lower()
            src_words = src_line.split() + ['<SEP>'] # TODO <EOS>
            tgt_words = tgt_line.split() + ['<EOS>']
            # if len(src_words) < 7: # TODO
            train_data.append((src_words, tgt_words))
        selection  = train_data[:]
        train_data = selection[:10000]
        dev_data   = selection[10000:12000]

        vocab_size, hidden_size, epoch_count = 15000, 512, 30 # TODO

        src_count, tgt_count = Counter(), Counter()
        src_vocab, tgt_vocab = Vocab(), Vocab()
        for src_words, tgt_words in train_data:
            for word in src_words:
                src_count[word] += 1
            for word in tgt_words:
                tgt_count[word] += 1
        for word, _ in src_count.most_common(vocab_size):
            src_vocab.add(word)
        for word, _ in tgt_count.most_common(vocab_size):
            tgt_vocab.add(word)
        # for src_words, tgt_words in train_data:
        #     src_vocab |= src_words
        #     tgt_vocab |= tgt_words

        model = Model(src_vocab, tgt_vocab, hidden_size).to(device)

        # if not args.dev:
        #     raise ValueError('--dev required')
        # dev_data = []
        # with open('tmp_file.txt', 'w') as f:
        #     for src_words, tgt_words in dev_data:
        #         f.write("%s <SEP> %s\n" % (src_words, tgt_words))
        # for line in open(args.dev):
        #     lsplit = line.split('\t')
        #     src_line, tgt_line = lsplit[0], lsplit[1]
        #     src_words = src_line.split() + ['<EOS>']
        #     tgt_words = tgt_line.split() + ['<EOS>']
        #     dev_data.append((src_words, tgt_words))

    elif args.load:
        if args.save:
            raise ValueError('--save can only be used with --train')
        if args.dev:
            raise ValueError('--dev can only be used with --train')
        model = torch.load(args.load)

    else: raise ValueError('either --train or --load required')

    if args.infile and not args.outfile:
        raise ValueError('-o required')

    if args.train:
        opt = torch.optim.Adam(model.parameters(), lr=0.0003) # TODO lr=1e-4

        best_dev_loss = None
        for epoch in range(epoch_count):
            random.shuffle(train_data)

            train_loss, train_tgt_words = 0., 0.
            for src_words, tgt_words in tqdm.tqdm(train_data):
                loss = model(src_words, tgt_words)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                train_tgt_words += len(tgt_words)
            
            dev_loss, dev_tgt_words = 0., 0.
            X, Y = [], []
            for line_num, (src_words, tgt_words) in enumerate(dev_data):
                dev_loss += model(src_words, tgt_words).item()
                dev_tgt_words += len(tgt_words)
                # if line_num < 1:
                #     translation = model.translate(src_words)
                #     print('>', ' '.join(tgt_words[:-1]))
                #     print('<', ' '.join(translation))
                translation = model.translate(src_words)
                X.append(translation)
                Y.append([tgt_words[:-1]])

            print(f'[{epoch + 1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_tgt_words)} dev_loss={dev_loss} dev_ppl={math.exp(dev_loss/dev_tgt_words)} dev_bleu={bleu_score(X, Y)}', flush=True)

            if best_dev_loss is None or dev_loss < best_dev_loss:
                print('saving best model...')
                best_model = copy.deepcopy(model)
                if args.save:
                    torch.save(model, args.save)
                best_dev_loss = dev_loss
            print()

        model = best_model

        # with open(args.outfile, 'w') as outfile:
        #     test_data = []
        #     for line in open(args.infile):
        #         lsplit = line.split('\t')
        #         src_line, tgt_line = lsplit[0].lower(), lsplit[1].lower()
        #         src_words = src_line.split() + ['<SEP>']
        #         tgt_words = tgt_line.split()
        #         if len(src_words) < 5:
        #             test_data.append((src_words, tgt_words))
        #     X, Y = [], []
        #     for src_words, tgt_words in test_data:
        #         translation = model.translate(src_words)
        #         X.append(translation)
        #         Y.append([tgt_words])
        #         print(' '.join(tgt_words) + '\n' + ' '.join(translation) + '\n', file=outfile)
        #     print(f'bleu_score={bleu_score(X, Y)}')

    elif args.infile:
        with open(args.outfile, 'w') as outfile:
            test_data = []
            for line in open(args.infile):
                words = line.lower().split() + ['<SEP>']
                # if len(words) < 7: # TODO
                #     test_data.append(words)
            for words in test_data:
                translation = model.translate(words)
                print(' '.join(translation), file=outfile)
