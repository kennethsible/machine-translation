import torch, random, copy, math, time, tqdm, re
from sacremoses import MosesTokenizer, MosesDetokenizer
from sacrebleu.metrics import BLEU, CHRF
from subword_nmt.apply_bpe import BPE
from datetime import timedelta
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bleu, chrf = BLEU(), CHRF()

def tokenize(input, src_lang):
    return MosesTokenizer(src_lang).tokenize(input, return_str=True)

def detokenize(output, tgt_lang):
    output = MosesDetokenizer(tgt_lang).detokenize(output)
    return re.sub('(@@ )|(@@ ?$)', '', output)

def subsequent_mask(size):
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Linear(nn.Module):

    def __init__(self, d_inp, d_out):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(d_out, d_inp))
        self.bias = torch.nn.Parameter(torch.empty(d_out))
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.normal_(self.bias, std=0.01)

    def forward(self, x):
        return x @ self.weight.transpose(0, 1) + self.bias

class LogSoftmax(nn.Module):

    def __init__(self, d_model, vocab):
        super(LogSoftmax, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab, d_model))
        torch.nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        # https://aclanthology.org/N18-1031
        weight = nn.functional.normalize(self.weight, dim=-1)
        return torch.log_softmax(x @ weight.transpose(0, 1), dim=-1)

class Embedding(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab, d_model))
        torch.nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        # https://aclanthology.org/N18-1031
        return nn.functional.normalize(self.weight[x], dim=-1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        enc = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('enc', enc.unsqueeze(0))

    def forward(self, x):
        x = x + self.enc[:, : x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ff_1 = Linear(d_model, d_ff)
        self.ff_2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.ff_2(self.dropout(self.ff_1(x).relu()))

class LayerNorm(nn.Module):

    def __init__(self, size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean, var = torch.mean(x, dim=-1, keepdim=True), torch.var(x, dim=-1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class ScaleNorm(nn.Module):
    # https://aclanthology.org/2019.iwslt-1.17

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(scale)
        self.eps = eps

    def forward(self, x):
        return self.scale * nn.functional.normalize(x, dim=-1, eps=self.eps)

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clone(Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None: # TODO Score Caching
            scores = scores.masked_fill(mask == 0, -torch.inf)
        scores = torch.softmax(scores, dim=-1)
        if dropout: scores = dropout(scores)
        return scores @ value

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        query, key, value = [
            linear(x).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        x = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).reshape(n_batches, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)

class SublayerConnection(nn.Module):

    def __init__(self, scale, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(scale)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(n_heads, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        scale = torch.tensor(d_model, dtype=torch.float32)
        self.sublayers = clone(SublayerConnection(torch.sqrt(scale), dropout), 2)

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_att(x, x, x, mask))
        return self.sublayers[1](x, self.ff)

class Encoder(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, dropout, N):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(d_model, d_ff, n_heads, dropout), N)
        scale = torch.tensor(d_model, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_att  = MultiHeadAttention(n_heads, d_model)
        self.cross_att = MultiHeadAttention(n_heads, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        scale = torch.tensor(d_model, dtype=torch.float32)
        self.sublayers = clone(SublayerConnection(torch.sqrt(scale), dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_att(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.cross_att(x, m, m, src_mask))
        return self.sublayers[2](x, self.ff)

class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, n_heads, dropout, N):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(d_model, d_ff, n_heads, dropout), N)
        scale = torch.tensor(d_model, dtype=torch.float32)
        self.norm = ScaleNorm(torch.sqrt(scale))

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Model(nn.Module):

    def __init__(self, vocab_size, d_model=512, d_ff=2048, n_heads=8, dropout=0.1, N=6):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(d_model, d_ff, n_heads, dropout, N)
        self.decoder = Decoder(d_model, d_ff, n_heads, dropout, N)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.src_embed = nn.Sequential(Embedding(d_model, vocab_size), PositionalEncoding(d_model, dropout))
        self.tgt_embed = nn.Sequential(Embedding(d_model, vocab_size), PositionalEncoding(d_model, dropout))
        self.generator = LogSoftmax(d_model, vocab_size)

    def forward(self, src_nums, tgt_nums, src_mask, tgt_mask):
        return self.decode(self.encode(src_nums, src_mask), src_mask, tgt_nums, tgt_mask)

    def encode(self, src_nums, src_mask):
        return self.encoder(self.src_embed(src_nums), src_mask)

    def decode(self, memory, src_mask, tgt_nums, tgt_mask):
        return self.decoder(self.tgt_embed(tgt_nums), memory, src_mask, tgt_mask)

class LabelSmoothing(nn.Module):

    def __init__(self, smoothing):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        l = torch.gather(input, dim=-1, index=target.unsqueeze(-1))
        return (self.smoothing - 1) * l.sum() - self.smoothing * input.mean()

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

class Batch:

    def __init__(self, src_nums, tgt_nums, padding_idx):
        self.src_nums = src_nums
        self.tgt_nums = tgt_nums
        self.src_mask = (self.src_nums != padding_idx).unsqueeze(-2)
        self.tgt_mask = (self.tgt_nums != padding_idx).unsqueeze(-2)
        self.n_tokens = self.tgt_mask.detach().sum()
        self.tgt_mask = self.tgt_mask & subsequent_mask(self.tgt_nums.size(-1))

    @staticmethod
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

class EarlyStopping:

    def __init__(self, patience, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0

    def __call__(self, curr_loss, prev_loss):
        if (curr_loss - prev_loss) > self.min_delta:
            self.count += 1
            if self.count >= self.patience:
                return True

def train_epoch(data, model, vocab, criterion, optimizer=None, mode='train'):
    total_loss = 0
    for batch in tqdm.tqdm(data):
        src_words, tgt_words = zip(*batch)
        src_nums = torch.stack([vocab.numberize(*words) for words in src_words])
        tgt_nums = torch.stack([vocab.numberize(*words) for words in tgt_words])
        batch = Batch(src_nums, tgt_nums[:, :-1], vocab.padding_idx)

        logits = model(batch.src_nums, batch.tgt_nums, batch.src_mask, batch.tgt_mask)
        lprobs = model.generator(logits)
        loss = criterion(torch.flatten(lprobs, 0, 1), torch.flatten(tgt_nums[:, 1:]))
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() / batch.n_tokens
    return total_loss

def greedy_search(model, memory, src_mask=None, max_len=256, start_word=0, end_word=1):
    path = torch.full((1, 1), start_word, device=device)
    for i in range(1, max_len + 1):
        logits = model.decode(memory, src_mask, path, subsequent_mask(i))
        lprobs = model.generator(logits[:, -1])
        next_word = torch.argmax(lprobs, dim=-1)
        if next_word == end_word: break
        path = torch.cat([path, next_word.unsqueeze(0)], dim=-1)
    return path.squeeze(0)[1:]

def beam_search(model, memory, beam_size, src_mask=None, max_len=256, start_word=0, end_word=1):
    probs = torch.zeros(beam_size, device=device)
    paths = torch.full((beam_size, max_len + 1), start_word, device=device)

    complete = [] # TODO Length Normalization
    for i in range(1, max_len + 1):
        logits = model.decode(memory.expand(beam_size, -1, -1),
            src_mask, paths[:, :i], subsequent_mask(i))
        lprobs = model.generator(logits[:, -1])

        hypotheses = torch.add(probs.unsqueeze(1), lprobs).flatten()
        topv, topi = torch.topk(hypotheses, beam_size)
        probs, paths = topv, paths[torch.trunc(topi / model.vocab_size).long()]
        paths[:, i] = torch.remainder(topi, model.vocab_size)

        finished = paths[:, i] == end_word
        complete.extend(zip(probs[finished], paths[finished, :i]))
        probs, paths = probs[~finished], paths[~finished]
        if paths.size(0) < beam_size:
            beam_size = paths.size(0)
        if beam_size == 0: break

    if paths.size(0) > 0:
        complete.extend(zip(probs, paths))
    complete.sort(key=lambda state: -state[0])
    return [path[1:] for _, path in complete]

def train_model(train_file, val_file, vocab_file, model_file, tgt_lang, config):
    train_data, val_data = [], []
    for data, data_file in ((train_data, train_file), (val_data, val_file)):
        data[:] = Batch.load_data(data_file, config['data_limit'],
            config['batch_size'], config['max_len'])
    assert 0 < len(val_data) < len(train_data)

    vocab = Vocab()
    with open(vocab_file) as file:
        for line in file:
            vocab.add(line.split()[0])
    open('DEBUG.log', 'w').close()

    model = Model(vocab.size()).to(device)
    model.src_embed[0].weight = model.tgt_embed[0].weight
    model.generator.weight = model.tgt_embed[0].weight

    criterion = LabelSmoothing(config['smoothing'])
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    stopping = EarlyStopping(config['patience'])

    best_score, prev_loss = 0, torch.inf
    for epoch in range(config['n_epochs']):
        random.shuffle(train_data)
    
        start = time.time()
        model.train()
        train_loss = train_epoch(
            train_data,
            model,
            vocab,
            criterion,
            optimizer,
            mode='train'
        )

        model.eval()
        with torch.no_grad():
            val_loss = train_epoch(
                val_data,
                model,
                vocab,
                criterion,
                mode='eval',
            )
        elapsed = timedelta(seconds=(time.time() - start))

        with open('DEBUG.log', 'a') as file:
            output = f'[{epoch + 1}] Train Loss: {train_loss} | Validation Loss: {val_loss}'
            print(output, flush=True)
            file.write(output + f' | Train Time: {elapsed}\n')

        start = time.time()
        candidate, reference = [], []
        with torch.no_grad():
            for batch in val_data:
                for src_words, tgt_words in batch:
                    src_nums = vocab.numberize(*src_words).unsqueeze(0)
                    tgt_nums = vocab.numberize(*tgt_words).unsqueeze(0)
                    batch = Batch(src_nums, tgt_nums, vocab.padding_idx)

                    memory = model.encode(batch.src_nums, batch.src_mask)
                    model_out = greedy_search(model, memory, batch.src_mask) if config['beam_size'] is None \
                        else beam_search(model, memory, config['beam_size'], batch.src_mask)[0]
                    reference.append(detokenize([vocab.denumberize(x)
                        for x in batch.tgt_nums[0] if x != vocab.padding_idx], tgt_lang))
                    candidate.append(detokenize(vocab.denumberize(*model_out), tgt_lang))

        bleu_score = bleu.corpus_score(candidate, [reference])
        chrf_score = chrf.corpus_score(candidate, [reference])
        elapsed = timedelta(seconds=(time.time() - start))

        with open('DEBUG.log', 'a') as file:
            output = f'  {chrf_score} | {bleu_score}'
            print(output, flush=True)
            file.write(output + f' | Decode Time: {elapsed}\n')
            if bleu_score.score > best_score:
                print('Saving Model...', flush=True)
                file.write('Saving Model...\n')
                torch.save(model.state_dict(), model_file)
                best_score = bleu_score.score
            if stopping(val_loss, prev_loss):
                print('Stopping Early...', flush=True)
                file.write('Stopping Early...\n')
                break
            print()
        prev_loss = val_loss

def score_model(test_file, vocab_file, model_file, out_file, tgt_lang, config):
    test_data = Batch.load_data(test_file)

    vocab = Vocab()
    with open(vocab_file) as file:
        for line in file:
            vocab.add(line.split()[0])

    model = Model(vocab.size()).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    start = time.time()
    candidate, reference = [], []
    with torch.no_grad():
        for src_words, tgt_words in tqdm.tqdm(test_data):
            src_nums = vocab.numberize(*src_words).unsqueeze(0)
            tgt_nums = vocab.numberize(*tgt_words).unsqueeze(0)
            batch = Batch(src_nums, tgt_nums, vocab.padding_idx)

            memory = model.encode(batch.src_nums, batch.src_mask)
            model_out = greedy_search(model, memory, batch.src_mask) if config['beam_size'] is None \
                else beam_search(model, memory, config['beam_size'], batch.src_mask)[0]
            reference.append(detokenize([vocab.denumberize(x)
                for x in batch.tgt_nums[0] if x != vocab.padding_idx], tgt_lang))
            candidate.append(detokenize(vocab.denumberize(*model_out), tgt_lang))

    bleu_score = bleu.corpus_score(candidate, [reference])
    chrf_score = chrf.corpus_score(candidate, [reference])
    elapsed = timedelta(seconds=(time.time() - start))

    with open(out_file, 'w') as file:
        output = f'{chrf_score} | {bleu_score}'
        print(output, flush=True)
        file.write(output + f' | Decode Time: {elapsed}\n\n')
        for translation in candidate:
            file.write(translation + '\n')

def translate(input, vocab_file, codes_file, model_file, src_lang, tgt_lang, config):
    with open(codes_file) as file:
        input = BPE(file).process_line(tokenize(input, src_lang))
    words = ['<BOS>'] + input.split() + ['<EOS>']

    vocab = Vocab()
    with open(vocab_file) as file:
        for line in file:
            vocab.add(line.split()[0])

    model = Model(vocab.size()).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    with torch.no_grad():
        src_nums = vocab.numberize(*words).unsqueeze(0)
        memory = model.encode(src_nums, src_mask=None)
        model_out = greedy_search(model, memory) if config['beam_size'] is None \
            else beam_search(model, memory, config['beam_size'])[0]
    return detokenize(vocab.denumberize(*model_out), tgt_lang)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='random seed')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommands')

    train_parser = subparsers.add_parser('train', help='train model')
    train_parser.add_argument('--data', metavar='FILE', required=True, type=str, help='training data')
    train_parser.add_argument('--val', metavar='FILE', required=True, type=str, help='validation data')
    train_parser.add_argument('--langs', nargs=2, metavar='LANG', required=True, type=str, help='source/target language')
    train_parser.add_argument('--vocab', metavar='FILE', required=True, type=str, help='shared vocabulary')
    train_parser.add_argument('--save', metavar='FILE', type=str, help='save state_dict')

    score_parser = subparsers.add_parser('score', help='score model')
    score_parser.add_argument('--data', metavar='FILE', required=True, type=str, help='test data')
    score_parser.add_argument('--langs', nargs=2, metavar='LANG', required=True, type=str, help='source/target language')
    score_parser.add_argument('--vocab', metavar='FILE', required=True, type=str, help='shared vocabulary')
    score_parser.add_argument('--load', metavar='FILE', required=True, type=str, help='load state_dict')
    score_parser.add_argument('--out', metavar='FILE', required=True, type=str, help='save score/output')

    input_parser = subparsers.add_parser('input', help='translate input')
    input_parser.add_argument('--langs', nargs=2, metavar='LANG', required=True, type=str, help='source/target language')
    input_parser.add_argument('--vocab', metavar='FILE', required=True, type=str, help='shared vocabulary')
    input_parser.add_argument('--codes', metavar='FILE', required=True, type=str, help='BPE codes file')
    input_parser.add_argument('--load', metavar='FILE', required=True, type=str, help='load state_dict')
    input_parser.add_argument('string', type=str, help='input string')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    config = {
        'n_epochs':     25,
        'lr':           3e-4,
        'smoothing':    0.1,
        'beam_size':    5,
        'patience':     2,
        'data_limit':   None,
        'batch_size':   32,
        'max_len':      128
    }

    if 'train' in args.subcommands:
        train_model(args.data, args.val, args.vocab, args.save, args.langs[1], config)
    elif 'score' in args.subcommands:
        score_model(args.data, args.vocab, args.load, args.out, args.langs[1], config)
    elif 'input' in args.subcommands:
        print(translate(args.string, args.vocab, args.codes, args.load, *args.langs, config))
