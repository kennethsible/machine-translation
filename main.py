from manager import Manager, Tokenizer
from model import EarlyStopping, train_epoch
from layers import CrossEntropy
from score import score_model
from datetime import timedelta
import torch, random, time, tqdm, toml

def train_model(manager, tokenizer, model_file=None, *, feedback=False):
    if model_file:
        manager.model.src_embed[0].weight = manager.model.tgt_embed[0].weight
        manager.model.generator.weight = manager.model.tgt_embed[0].weight

    criterion = CrossEntropy(manager.config['smoothing'])
    optimizer = torch.optim.Adam(manager.model.parameters(), manager.config['lr'])
    stopping = EarlyStopping(manager.config['patience'], manager.config['min-delta'])

    best_score, prev_loss = 0, torch.inf
    for epoch in range(manager.config['n-epochs']):
        random.shuffle(manager.data)
    
        start = time.perf_counter()
        manager.model.train()
        train_loss = train_epoch(
            tqdm.tqdm(manager.data) if feedback else manager.data,
            manager.model,
            criterion,
            optimizer,
            mode='train'
        )

        manager.model.eval()
        with torch.no_grad():
            val_loss = train_epoch(
                manager.test,
                manager.model,
                criterion,
                optimizer=None,
                mode='eval',
            )
        elapsed = timedelta(seconds=(time.perf_counter() - start))

        status_update = f'[{epoch + 1}] Train Loss = {train_loss} | Val Loss = {val_loss} | Train Time: {elapsed}'
        if model_file:
            with open(model_file + '.log', 'a') as file:
                file.write(status_update + '\n')
        print(status_update)

        bleu_score, _, _ = score_model(manager, tokenizer, model_file, indent=((epoch + 1) // 10 + 4))

        if bleu_score.score > best_score:
            if model_file:
                with open(model_file + '.log', 'a') as file:
                    file.write('Saving Model...\n')
            print('Saving Model...')
            manager.save_model(model_file)
            best_score = bleu_score.score
        if stopping(val_loss, prev_loss):
            if model_file:
                with open(model_file + '.log', 'a') as file:
                    file.write('Stopping Early...\n')
            print('Stopping Early...')
            break
        if model_file:
            with open(model_file + '.log', 'a') as file:
                file.write('\n')
        print()

        prev_loss = val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    parser.add_argument('--data', metavar='FILE', help='training data')
    parser.add_argument('--test', metavar='FILE', help='validation data')
    parser.add_argument('--vocab', metavar='FILE', help='shared vocab')
    parser.add_argument('--config', metavar='FILE', default='model.config', help='model config')
    parser.add_argument('--load', metavar='FILE', help='load state_dict')
    parser.add_argument('--save', metavar='FILE', help='save state_dict')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--tqdm', action='store_true', help='toggle tqdm')
    args, unknown = parser.parse_known_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    src_lang, tgt_lang = args.lang
    with open(args.config) as file:
        config = toml.load(file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and arg[2:] in config:
            if len(unknown) >= i + 1:
                config[arg[2:]] = int(unknown[i + 1])

    if not args.data:
        args.data = f'data/training/train.tok.bpe.{src_lang}{tgt_lang}'
    if not args.test:
        args.test = f'data/validation/val.tok.bpe.{src_lang}{tgt_lang}'
    if not args.vocab:
        args.vocab = f'data/vocab.{src_lang}{tgt_lang}'
    if not args.save:
        args.save = f'data/model.{src_lang}{tgt_lang}'
        open(args.save + '.log', 'w').close()

    manager = Manager(
        src_lang,
        tgt_lang,
        config,
        device,
        args.data,
        args.test,
        args.vocab
    )
    if args.load:
        manager.load_model(args.load)
    tokenizer = Tokenizer(src_lang, tgt_lang)

    train_model(manager, tokenizer, args.save, feedback=args.tqdm)

if __name__ == '__main__':
    import argparse
    main()
