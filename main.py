from model import Model, EarlyStopping
from decoding import greedy_search, beam_search
from manager import load_data, Vocab, Batch
from sacrebleu.metrics import BLEU, CHRF
from layers import LabelSmoothing
from translate import detokenize
from datetime import timedelta
import torch, random, tqdm, time, json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bleu, chrf = BLEU(), CHRF()

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

def train_model(train_file, val_file, vocab_file, model_file, tgt_lang, config):
    train_data, val_data = [], []
    for data, data_file in ((train_data, train_file), (val_data, val_file)):
        data[:] = load_data(data_file, config['data_limit'],
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
                        else beam_search(model, memory, config['beam_size'], batch.src_mask)
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
    test_data = load_data(test_file)

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
                else beam_search(model, memory, config['beam_size'], batch.src_mask)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='random seed')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommands')

    train_parser = subparsers.add_parser('train', help='train model')
    train_parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    train_parser.add_argument('--data', nargs=2, metavar='FILE', help='training/validation data')
    train_parser.add_argument('--vocab', metavar='FILE', help='vocab (from BPE)')
    train_parser.add_argument('--config', metavar='FILE', default='config.json', help='model config')
    train_parser.add_argument('--save', metavar='FILE', help='save state_dict')

    score_parser = subparsers.add_parser('score', help='score model')
    score_parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    score_parser.add_argument('--data', metavar='FILE', help='testing data')
    score_parser.add_argument('--vocab', metavar='FILE', help='vocab (from BPE)')
    score_parser.add_argument('--config', metavar='FILE', default='config.json', help='model config')
    score_parser.add_argument('--load', metavar='FILE', help='load state_dict')
    score_parser.add_argument('--out', metavar='FILE', default='test.out', help='store output')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    src_lang, tgt_lang = args.lang
    if not args.vocab:
        args.vocab = f'data/vocab.{src_lang}{tgt_lang}'

    with open(args.config, 'r') as file:
        config = json.load(file)

    if 'train' in args.subcommands:
        if not args.data:
            args.data = (
                f'data/training/train.tok.bpe.{src_lang}{tgt_lang}',
                f'data/validation/val.tok.bpe.{src_lang}{tgt_lang}'
            )
        if not args.save:
            args.save = f'model.{src_lang}{tgt_lang}'
        train_model(*args.data, args.vocab, args.save, tgt_lang, config)
    elif 'score' in args.subcommands:
        if not args.data:
            args.data = f'data/testing/test.tok.bpe.{src_lang}{tgt_lang}'
        if not args.load:
            args.load = f'model.{src_lang}{tgt_lang}'
        if not args.out:
            args.out =  f'test.out'
        score_model(args.data, args.vocab, args.load, args.out, tgt_lang, config)
