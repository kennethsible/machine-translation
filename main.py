from model import Model, EarlyStopping
from decoding import beam_search
from manager import device, load_data, Vocab, Batch
from sacrebleu.metrics import BLEU, CHRF
from layers import LabelSmoothing
from translate import detokenize
from datetime import timedelta
import torch, random, tqdm, time, json

bleu, chrf = BLEU(), CHRF()

def train_epoch(data, model, criterion, optimizer=None, mode='train'):
    total_loss = 0
    for batch in tqdm.tqdm(data):
        logits = model(batch.src_nums, batch.tgt_nums[:, :-1], batch.src_mask, batch.tgt_mask)
        lprobs = model.generator(logits)
        loss = criterion(torch.flatten(lprobs, 0, 1), torch.flatten(batch.tgt_nums[:, 1:]))
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() / batch.n_tokens
    return total_loss

def train_model(train_file, val_file, vocab_file, model_file, log_file, tgt_lang, config):
    vocab = Vocab()
    with open(vocab_file) as file:
        for line in file:
            vocab.add(line.split()[0])
    assert vocab.size() > 0

    train_data, val_data = [], []
    for data, data_file in ((train_data, train_file), (val_data, val_file)):
        data[:] = load_data(data_file, vocab, config['data_limit'], config['max_len'], config['batch_size'])
    assert len(train_data) > 0 and len(val_data) > 0

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
            criterion,
            optimizer,
            mode='train'
        )

        model.eval()
        with torch.no_grad():
            val_loss = train_epoch(
                val_data,
                model,
                criterion,
                mode='eval',
            )
        elapsed = timedelta(seconds=(time.time() - start))

        with open(log_file, 'a') as file:
            output = f'[{epoch + 1}] Train Loss: {train_loss} | Validation Loss: {val_loss}'
            print(output, flush=True)
            file.write(output + f' | Train Time: {elapsed}\n')

        _, bleu_score, chrf_score = score_model(val_data, vocab, model, log_file, tgt_lang, config)

        with open(log_file, 'a') as file:
            if bleu_score.score > best_score:
                print('Saving Model...', flush=True)
                file.write('Saving Model...\n')
                torch.save(model.state_dict(), model_file)
                best_score = bleu_score.score
            if stopping(val_loss, prev_loss):
                print('Stopping Early...', flush=True)
                file.write('Stopping Early...\n')
                break
            print(flush=True)
            file.write('\n')
        prev_loss = val_loss

def score_model(test_file, vocab_file, model_file, out_file, tgt_lang, config):
    if isinstance(vocab_file, Vocab):
        vocab = vocab_file
    else:
        vocab = Vocab()
        with open(vocab_file) as file:
            for line in file:
                vocab.add(line.split()[0])
        assert vocab.size() > 0

    if isinstance(test_file, list):
        test_data = test_file
        unbatched = []
        for batch in test_data:
            for i in range(batch.size()):
                unbatched.append((batch.src_nums[i], batch.tgt_nums[i]))
        test_data = unbatched
    else:
        test_data = load_data(test_file, vocab, config['data_limit'])
        assert len(test_data) > 0

    if isinstance(model_file, Model):
        model = model_file
    else:
        model = Model(vocab.size()).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()

    start = time.time()
    candidate, reference = [], []
    with torch.no_grad():
        for src_nums, tgt_nums in test_data:
            batch = Batch(src_nums, tgt_nums, vocab.padding_idx)
            memory = model.encode(batch.src_nums, batch.src_mask)
            model_out = beam_search(model, memory, config['beam_size'], batch.src_mask)

            reference.append(detokenize([vocab.denumberize(x)
                for x in batch.tgt_nums[0] if x != vocab.padding_idx], tgt_lang))
            candidate.append(detokenize(vocab.denumberize(*model_out), tgt_lang))

    bleu_score = bleu.corpus_score(candidate, [reference])
    chrf_score = chrf.corpus_score(candidate, [reference])
    elapsed = timedelta(seconds=(time.time() - start))

    with open(out_file, 'a') as file:
        output = f'{chrf_score} | {bleu_score}'
        print(output, flush=True)
        file.write(output + f' | Decode Time: {elapsed}\n')

    return candidate, bleu_score, chrf_score

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
    train_parser.add_argument('--out', metavar='FILE', help='store log file')

    score_parser = subparsers.add_parser('score', help='score model')
    score_parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    score_parser.add_argument('--data', metavar='FILE', help='testing data')
    score_parser.add_argument('--vocab', metavar='FILE', help='vocab (from BPE)')
    score_parser.add_argument('--config', metavar='FILE', default='config.json', help='model config')
    score_parser.add_argument('--load', metavar='FILE', help='load state_dict')
    score_parser.add_argument('--out', metavar='FILE', help='store output')
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
        if not args.out:
            args.out =  f'log.{src_lang}{tgt_lang}'
        train_model(*args.data, args.vocab, args.save, args.out, tgt_lang, config)
    elif 'score' in args.subcommands:
        if not args.data:
            args.data = f'data/testing/test.tok.bpe.{src_lang}{tgt_lang}'
        if not args.load:
            args.load = f'model.{src_lang}{tgt_lang}'
        if not args.out:
            args.out =  f'out.{src_lang}{tgt_lang}'
        candidate, _, _ = score_model(args.data, args.vocab, args.load, args.out, tgt_lang, config)
        with open(args.out, 'a') as file:
            file.write('\n')
            for translation in candidate:
                file.write(translation + '\n')
