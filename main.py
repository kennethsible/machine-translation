from manager import Manager, Tokenizer
from score import score_model
from datetime import timedelta
import torch, random, time, tqdm, toml

def train_epoch(data, model, criterion, optimizer=None, *, mode='train'):
    total_loss = 0.
    for batch in data:
        src_nums, src_mask = batch.src_nums, batch.src_mask
        tgt_nums, tgt_mask = batch.tgt_nums, batch.tgt_mask
        logits = model(src_nums, src_mask, tgt_nums[:, :-1], tgt_mask)
        loss = criterion(torch.flatten(logits, 0, 1), torch.flatten(tgt_nums[:, 1:]))

        if optimizer and mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() / batch.num_tokens
        del logits, loss
    return total_loss

def train_model(manager, tokenizer, model_file=None, feedback=False):
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=manager.config['label_smoothing'])
    optimizer = torch.optim.Adam(manager.model.parameters(), lr=manager.config['lr'])

    best_loss, patience = torch.inf, 0
    for epoch in range(manager.config['max_epochs']):
        if epoch > 0:
            if model_file:
                with open(model_file + '.log', 'a') as file:
                    file.write('\n')
            print()
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

        checkpoint = f'[{str(epoch + 1).rjust(len(str(manager.config["max_epochs"])), "0")}]'
        checkpoint += f' Training PPL = {torch.exp(train_loss):.4e}'
        checkpoint += f' | Validation PPL = {torch.exp(val_loss):.4e}'
        checkpoint += f' | Elapsed Time = {elapsed}'
        if model_file:
            with open(model_file + '.log', 'a') as file:
                file.write(checkpoint + '\n')
        print(checkpoint)

        if val_loss < best_loss:
            manager.save_model(model_file)
            if model_file:
                with open(model_file + '.log', 'a') as file:
                    file.write('Saving Model.\n')
            print('Saving Model.')
            best_loss, patience = val_loss, 0
        else:
            patience += 1

        if patience >= manager.config['patience']:
            if model_file:
                with open(model_file + '.log', 'a') as file:
                    file.write('Patience Reached for Early Stopping.\n')
            print('Patience Reached for Early Stopping.')
            return score_model(manager, tokenizer, model_file, indent=((epoch + 1) // 10 + 4))

    if model_file:
        with open(model_file + '.log', 'a') as file:
            file.write('Maximum Number of Epochs Reached.\n')
    print('Maximum Number of Epochs Reached.')
    return score_model(manager, tokenizer, model_file, indent=((epoch + 1) // 10 + 4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, required=True, help='source/target language')
    parser.add_argument('--data', metavar='FILE', help='training data')
    parser.add_argument('--test', metavar='FILE', help='validation data')
    parser.add_argument('--vocab', metavar='FILE', help='shared vocab')
    parser.add_argument('--config', metavar='FILE', help='model config')
    parser.add_argument('--load', metavar='FILE', help='load state_dict')
    parser.add_argument('--save', metavar='FILE', help='save state_dict')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--tqdm', action='store_true', help='toggle tqdm')
    args, unknown = parser.parse_known_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    src_lang, tgt_lang = args.lang
    if not args.config:
        args.config = 'model.config'
    with open(args.config) as file:
        config = toml.load(file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:], unknown[i + 1]
            config[option] = (int if value.isdigit() else float)(value)

    if not args.data:
        args.data = f'data/training/data.tok.bpe.{src_lang}{tgt_lang}'
    if not args.test:
        args.test = f'data/validation/data.tok.bpe.{src_lang}{tgt_lang}'
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
        args.vocab,
        args.data,
        args.test
    )
    if args.load:
        manager.load_model(args.load)
    tokenizer = Tokenizer(src_lang, tgt_lang)

    train_model(manager, tokenizer, args.save, args.tqdm)

if __name__ == '__main__':
    import argparse
    main()
