from model import Model, EarlyStopping
from manager import Vocab, load_data, device
from score import score_model
from layers import LabelSmoothing
from datetime import timedelta
import torch, random, time, json, tqdm

def iterator(iterable, progress):
    return tqdm.tqdm(iterable) if progress else iterable

def train_epoch(data, model, criterion, optimizer=None, progress=False, mode='train'):
    total_loss = 0
    for batch in iterator(data, progress):
        src_nums, tgt_nums = batch.src_nums, batch.tgt_nums
        src_mask, tgt_mask = batch.src_mask, batch.tgt_mask

        logits = model(src_nums, tgt_nums[:, :-1], src_mask, tgt_mask)
        lprobs = torch.flatten(model.generator(logits), 0, 1)
        loss = criterion(lprobs, torch.flatten(tgt_nums[:, 1:]))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() / batch.n_tokens
    return total_loss

def train_model(config):
    vocab = Vocab()
    with open(config['vocab']) as file:
        for line in file:
            vocab.add(line.split()[0])
    assert vocab.size() > 0

    train_data, val_data = [], []
    for data, data_file in ((train_data, config['data']), (val_data, config['test'])):
        data[:] = load_data(data_file, vocab, config['batch_size'], config['max_length'])
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
            config['tqdm'],
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

        print(f'[{epoch + 1}] Train Loss = {train_loss} | Val Loss = {val_loss} | Train Time: {elapsed}')

        bleu_score, _ = score_model(config | {
            'data': val_data,
            'vocab': vocab,
            'load': model,
            'out': None
        }, (epoch + 1) // 10 + 4)

        if bleu_score.score > best_score:
            print('Saving Model...')
            torch.save(model.state_dict(), config['save'])
            best_score = bleu_score.score
        if stopping(val_loss, prev_loss):
            print('Stopping Early...')
            break
        print(flush=True)

        prev_loss = val_loss

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    parser.add_argument('--data', metavar='FILE', help='training data')
    parser.add_argument('--test', metavar='FILE', help='validation data')
    parser.add_argument('--vocab', metavar='FILE', help='shared vocab')
    parser.add_argument('--config', metavar='FILE', default='model.config', help='model config')
    parser.add_argument('--save', metavar='FILE', help='save state_dict')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--tqdm', action='store_true', help='toggle tqdm')
    args, unknown = parser.parse_known_args()

    with open(args.config) as file:
        config = json.load(file)

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and arg[2:] in config:
            if len(unknown) >= i + 1:
                config[arg[2:]] = int(unknown[i + 1])

    src_lang, tgt_lang = args.lang
    config['lang'] = tgt_lang

    config['data'] = args.data if args.data \
        else f'data/training/train.tok.bpe.{src_lang}{tgt_lang}'
    config['test'] = args.test if args.test \
        else f'data/validation/val.tok.bpe.{src_lang}{tgt_lang}'
    config['vocab'] = args.vocab if args.vocab \
        else f'data/vocab.{src_lang}{tgt_lang}'
    config['save'] = args.save if args.save \
        else f'data/output/model.{src_lang}{tgt_lang}'
    config['tqdm'] = args.tqdm

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    train_model(config)
