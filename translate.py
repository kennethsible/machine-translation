import torch, random, math, copy, tqdm, re
from sacremoses import MosesDetokenizer
from sacrebleu.metrics import BLEU
import natlang as nl

bleu, md = BLEU(), MosesDetokenizer(lang='en')
device = torch.device('cuda')

# torch.manual_seed(0)

def detokenize(words):
    return re.sub('(@@ )|(@@ ?$)', '', md.detokenize(words))

class Encoder(torch.nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.emb = nl.Embedding(vocab_size, hidden_size)
        self.rnn = nl.LSTM(hidden_size, hidden_size, 2)

    def forward(self, nums):
        embs = self.emb(nums)
        return self.rnn(embs)

class Decoder(torch.nn.Module):

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.emb = nl.Embedding(vocab_size, hidden_size)
        self.rnn = nl.LSTMCell(hidden_size, hidden_size)
        self.tnh = nl.Tanh(2*hidden_size, hidden_size)
        self.out = nl.LogSoftmax(hidden_size, vocab_size)

    def start(self, H):
        batch_size = H[0].size()[0]
        h = self.rnn.h0.repeat(batch_size, 1)
        m = self.rnn.m0.repeat(batch_size, 1)
        return (m, h, H)

    def input(self, state, num):
        m, h, H = state
        emb = self.emb(num.unsqueeze(0))
        h, m = self.rnn(emb.squeeze(1), h, m)
        return (m, h, H)

    def output(self, state, mask=None):
        _, h, H = state
        HT = H.transpose(0, 1)
        c = nl.attention(h, HT, HT, mask)
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
        src_encs = self.encoder(src_nums.unsqueeze(0))
        mask = (src_nums == 0).unsqueeze(0)

        output, state = [], self.decoder.start(src_encs)
        for _ in range(100):
            out = self.decoder.output(state, mask)
            tgt_num = torch.argmax(out.squeeze(0)).view((1,))
            tgt_word = self.tgt_vocab.denumberize(tgt_num.item())
            if tgt_word == '<EOS>': break
            output.append(tgt_word)
            state = self.decoder.input(state, tgt_num)
        return output

    def forward(self, src_sents, tgt_sents):
        src_nums = torch.stack([self.src_vocab.numberize(*src_words)
            for src_words in src_sents]).to(device)
        src_encs = self.encoder(src_nums)
        mask = (src_nums == 0)

        seq_len = len(tgt_sents[0])
        loss, state = 0, self.decoder.start(src_encs)
        for i in range(seq_len):
            tgt_word = [tgt_words[i] for tgt_words in tgt_sents]
            out = self.decoder.output(state, mask)
            tgt_num = self.tgt_vocab.numberize(*tgt_word).to(device)
            loss -= sum(out[i, num] for i, num in enumerate(tgt_num) if num != 0)
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

    MAX_LEN = 20
    # VOCAB_SIZE = 10000
    BATCH_SIZE = 50
    EMBED_SIZE = 256
    LEARN_RATE = 1e-4
    EPOCH_COUNT = 20

    if args.train:
        training = []
        for line in open(args.train):
            src_line, tgt_line = line.split('\t')
            src_words = src_line.split() + ['<EOS>']
            tgt_words = tgt_line.split() + ['<EOS>']
            if not MAX_LEN or len(src_words) < MAX_LEN:
                training.append((src_words, tgt_words))

        MAX_TRAIN = 100000
        assert MAX_TRAIN < len(training)
        VAL_START = math.ceil(0.95 * MAX_TRAIN)
        training, validation = training[:VAL_START], training[VAL_START:MAX_TRAIN]

        batches = []
        training.sort(key=lambda x: len(x[0]))
        for i in range(BATCH_SIZE, len(training) + 1, BATCH_SIZE):
            batch = training[(i - BATCH_SIZE):i]
            src_max_len = max(len(src_words) for src_words, _ in batch)
            tgt_max_len = max(len(tgt_words) for _, tgt_words in batch)
            for src_words, tgt_words in batch:
                residual = src_max_len - len(src_words)
                if residual > 0:
                    src_words.extend(residual * ['<PAD>'])
                residual = tgt_max_len - len(tgt_words)
                if residual > 0:
                    tgt_words.extend(residual * ['<PAD>'])
            batches.append(batch)
        assert len(batches) == VAL_START//BATCH_SIZE
        training = batches

        batches = []
        validation.sort(key=lambda x: len(x[0]))
        for i in range(BATCH_SIZE, len(validation) + 1, BATCH_SIZE):
            batch = validation[(i - BATCH_SIZE):i]
            src_max_len = max(len(src_words) for src_words, _ in batch)
            tgt_max_len = max(len(tgt_words) for _, tgt_words in batch)
            for src_words, tgt_words in batch:
                residual = src_max_len - len(src_words)
                if residual > 0:
                    src_words.extend(residual * ['<PAD>'])
                residual = tgt_max_len - len(tgt_words)
                if residual > 0:
                    tgt_words.extend(residual * ['<PAD>'])
            batches.append(batch)
        assert len(batches) == (MAX_TRAIN - VAL_START)//BATCH_SIZE
        validation = batches

        # TODO use argparse to input vocab
        src_vocab, tgt_vocab = nl.Vocab(), nl.Vocab()
        with open('data/vocab.de') as vocab_file:
            for line in vocab_file.readlines(): # [:VOCAB_SIZE]
                src_vocab.add(line.split()[0])
        with open('data/vocab.en') as vocab_file:
            for line in vocab_file.readlines(): # [:VOCAB_SIZE]
                tgt_vocab.add(line.split()[0])

        model = Model(src_vocab, tgt_vocab, EMBED_SIZE).to(device)

    elif args.load:
        if args.save:
            raise ValueError('--save can only be used with --train')
        model = torch.load(args.load)

    else: raise ValueError('either --train or --load is required')

    if args.infile and not args.outfile:
        raise ValueError('-o is required')

    if args.train:
        opt = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

        best_loss = None
        for epoch in range(EPOCH_COUNT):
            random.shuffle(training)

            train_loss, total_words = 0., 0.
            for batch in tqdm.tqdm(training):
                loss = model(*zip(*batch))
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                # total_words += len(tgt_words)
            # train_ppl = math.exp(train_loss/total_words)

            candidate, reference = [], []

            dev_loss, total_words = 0., 0.
            for batch in validation:
                dev_loss += model(*zip(*batch)).item()
                for src_words, tgt_words in batch:
                    candidate.append(detokenize(model.translate(src_words)))
                    reference.append(detokenize(tgt_words[:tgt_words.index('<EOS>')]))
                    total_words += len(tgt_words)
                # total_words += len(tgt_words)
            # dev_ppl = math.exp(dev_loss/total_words)

            for x, y in zip(candidate[:5], reference[:5]):
                print(f'> {x}\n< {y}')
            score = bleu.corpus_score(candidate, [reference])
            # print(f'\n{score}\n')
            print()

            # print(f'[{epoch + 1}] train_loss={train_loss} train_ppl={train_ppl} dev_loss={dev_loss} dev_ppl={dev_ppl}\n{score}')
            print(f'[{epoch + 1}] train_loss={train_loss} dev_loss={dev_loss}\n{score}')
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
                words = line.lower().split() + ['<EOS>']
                if not MAX_LEN or len(words) < MAX_LEN:
                    test_data.append(words)
            for words in tqdm.tqdm(test_data):
                print(detokenize(model.translate(words)), file=outfile)
