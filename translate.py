import torch, random, math, copy, tqdm, re
from sacremoses import MosesDetokenizer
from sacrebleu.metrics import BLEU
import natlang as nl

bleu, md = BLEU(), MosesDetokenizer(lang='en')
device = torch.device('cuda')

def detokenize(words):
    return re.sub('(@@ )|(@@ ?$)', '', md.detokenize(words))

class Encoder(torch.nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.emb = nl.Embedding(vocab_size, hidden_size)
        self.rnn = nl.RNN(hidden_size, hidden_size, 2)

    def forward(self, src_nums):
        emb = self.emb(src_nums)
        H = self.rnn(emb)
        return H

class Decoder(torch.nn.Module):
    
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.emb = nl.Embedding(vocab_size, hidden_size)
        self.rnn = nl.RNNCell(hidden_size, hidden_size)
        self.tnh = nl.Tanh(2*hidden_size, hidden_size)
        self.out = nl.LogSoftmax(hidden_size, vocab_size)

    def start(self, H):
        h = self.rnn.h0
        # c = H[-1]
        # return (h, c)
        return (h, H)

    def input(self, state, tgt_num):
        h, H = state
        emb = self.emb(tgt_num).squeeze(0)
        h = self.rnn(emb, h)
        return (h, H)

    def output(self, state):
        h, H = state
        c = nl.attention(h, H, H)
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
        src_nums = self.src_vocab.numberize(*src_words).to(device)
        src_encs = self.encoder(src_nums)
        state = self.decoder.start(src_encs)
        tgt_words = []
        for _ in range(100):
            out = self.decoder.output(state)
            tgt_num = torch.argmax(out).view((1,))
            tgt_word = self.tgt_vocab.denumberize(tgt_num.item())
            if tgt_word == '<EOS>': break
            tgt_words.append(tgt_word)
            state = self.decoder.input(state, tgt_num)
        return tgt_words

    def forward(self, src_words, tgt_words):
        src_nums = self.src_vocab.numberize(*src_words).to(device)
        src_encs = self.encoder(src_nums)
        state = self.decoder.start(src_encs)
        loss = 0
        for tgt_word in tgt_words:
            out = self.decoder.output(state)
            tgt_num = self.tgt_vocab.numberize(tgt_word).to(device)
            loss -= out[tgt_num]
            state = self.decoder.input(state, tgt_num)
        return loss

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('infile', nargs='?', type=str, help='test data')
    parser.add_argument('-o', '--outfile', type=str, help='output file')
    parser.add_argument('--load', type=str, help='load model')
    parser.add_argument('--save', type=str, help='save model')
    args = parser.parse_args()

    seq_len, vocab_size, hidden_size, epoch_count, lr = 20, 10000, 256, 5, 1e-4

    if args.train:
        train_data = []
        for line in open(args.train):
            src_line, tgt_line = line.split('\t')
            src_words = src_line.split() + ['<SEP>']
            tgt_words = tgt_line.split() + ['<EOS>']
            if not seq_len or len(src_words) < seq_len:
                train_data.append((src_words, tgt_words))
        N = 10000 # TODO remove with minibatching
        i = math.ceil(0.8 * N)
        train_data, dev_data = train_data[:i], train_data[i:N]

        # TODO use argparse to input vocab
        src_vocab, tgt_vocab = nl.Vocab(), nl.Vocab()
        with open('data/vocab.de') as vocab_file:
            for line in vocab_file.readlines()[:vocab_size]:
                src_vocab.add(line.split()[0])
        with open('data/vocab.en') as vocab_file:
            for line in vocab_file.readlines()[:vocab_size]:
                tgt_vocab.add(line.split()[0])

        model = Model(src_vocab, tgt_vocab, hidden_size).to(device)

    elif args.load:
        if args.save:
            raise ValueError('--save can only be used with --train')
        model = torch.load(args.load)

    else: raise ValueError('either --train or --load is required')

    if args.infile and not args.outfile:
        raise ValueError('-o is required')

    if args.train:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best_loss = None
        for epoch in range(epoch_count):
            random.shuffle(train_data)

            train_loss, total_words = 0., 0.
            for src_words, tgt_words in tqdm.tqdm(train_data):
                loss = model(src_words, tgt_words)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                total_words += len(tgt_words)
            train_ppl = math.exp(train_loss/total_words)
            
            candidate, reference = [], []
            dev_loss, total_words = 0., 0.
            for line_num, (src_words, tgt_words) in enumerate(dev_data):
                dev_loss += model(src_words, tgt_words).item()
                total_words += len(tgt_words)
                candidate.append(detokenize(model.translate(src_words)))
                reference.append(detokenize(tgt_words[:-1]))
                # if line_num < 10: print(candidate[-1])
            dev_ppl = math.exp(dev_loss/total_words)
            score = bleu.corpus_score(candidate, [reference])

            print(f'[{epoch + 1}] train_loss={train_loss} train_ppl={train_ppl} dev_loss={dev_loss} dev_ppl={dev_ppl}\n{score}')
            if best_loss is None or dev_loss < best_loss:
                print('saving best model...')
                best_model = copy.deepcopy(model)
                if args.save:
                    torch.save(model, args.save)
                best_loss = dev_loss
            print()

        model = best_model

    elif args.infile:
        with open(args.outfile, 'w') as outfile:
            test_data = []
            for line in open(args.infile):
                words = line.lower().split() + ['<SEP>']
                if not seq_len or len(words) < seq_len:
                    test_data.append(words)
            for words in test_data:
                print(detokenize(model.translate(words)), file=outfile)
