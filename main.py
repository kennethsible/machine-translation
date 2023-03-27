from manager import Manager
from model import train_epoch
from datetime import timedelta
import torch, random, time, tqdm, toml

def train_model(manager, model_file=None, *, feedback=False):
    if model_file:
        manager.model.src_embed[0].weight = manager.model.tgt_embed[0].weight
        manager.model.generator.weight = manager.model.tgt_embed[0].weight

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=manager.config['label-smoothing'])
    optimizer = torch.optim.Adam(manager.model.parameters(), lr=manager.config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=manager.config['decay-factor'], patience=manager.config['patience'])

    prev_val_loss = torch.inf
    for epoch in range(manager.config['max-epochs']):
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
        scheduler.step(val_loss)
        elapsed = timedelta(seconds=(time.perf_counter() - start))

        checkpoint = f'[{str(epoch + 1).rjust(len(str(manager.config["max-epochs"])), "0")}]'
        checkpoint += f' Training PPL = {torch.exp(train_loss):.4e}'
        checkpoint += f' | Validation PPL = {torch.exp(val_loss):.4e}'
        checkpoint += f' | Learning Rate = {optimizer.param_groups[0]["lr"]:.4e}'
        checkpoint += f' | Elapsed Time = {elapsed}'
        if model_file:
            with open(model_file + '.log', 'a') as file:
                file.write(checkpoint + '\n')
        print(checkpoint)

        # score_model(manager, tokenizer, model_file, indent=((epoch + 1) // 10 + 4))

        if val_loss < prev_val_loss:
            manager.save_model(model_file)
            if model_file:
                with open(model_file + '.log', 'a') as file:
                    file.write('Saving Model Checkpoint.\n')
            print('Saving Model Checkpoint.')
            prev_val_loss = val_loss
        if optimizer.param_groups[0]['lr'] < manager.config['min-lr']:
            if model_file:
                with open(model_file + '.log', 'a') as file:
                    file.write('Minimum Learning Rate Reached. Stopping Training.\n')
            print('Minimum Learning Rate Reached. Stopping Training.')
            return

    if model_file:
        with open(model_file + '.log', 'a') as file:
            file.write('Maximum Number of Epochs Reached. Stopping Training.\n')
    print('Maximum Number of Epochs Reached. Stopping Training.')

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
        if arg[:2] == '--' and arg[2:] in config:
            if len(unknown) >= i + 1:
                try:
                    config[arg[2:]] = int(unknown[i + 1])
                except ValueError:
                    config[arg[2:]] = float(unknown[i + 1])

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

    train_model(manager, args.save, feedback=args.tqdm)

if __name__ == '__main__':
    import argparse
    main()
