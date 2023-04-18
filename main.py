from manager import Manager
from score import score_model
from datetime import timedelta
import random, torch, time, tqdm, toml

def train_epoch(manager, criterion, optimizer=None, scalar=None, visualize=False, *, mode='train'):
    data = manager.data if mode == 'train' else manager.test
    model, vocab = manager.model, manager.vocab
    decoding_size = vocab.size(decoding=True)

    total_loss, num_tokens = 0., 0
    for batch in tqdm.tqdm(data) if visualize else data:
        src_nums, src_mask = batch.src_nums, batch.src_mask
        tgt_nums, tgt_mask = batch.tgt_nums, batch.tgt_mask

        with torch.cuda.amp.autocast():
            logits = model(src_nums, src_mask, tgt_nums[:, :-1], tgt_mask, decoding_size)
            loss = criterion(torch.flatten(logits, 0, 1), torch.flatten(tgt_nums[:, 1:]))

        if optimizer and mode == 'train':
            optimizer.zero_grad()
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()

        total_loss += loss.item()
        num_tokens += batch.num_tokens
        del logits, loss

    return total_loss / num_tokens

def train_model(manager, logger, visualize=False):
    model, vocab = manager.model, manager.vocab
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.PAD,
        label_smoothing=manager.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=manager.lr)
    scalar = torch.cuda.amp.GradScaler()

    i, best_loss = 0, torch.inf
    for epoch in range(manager.max_epochs):
        random.shuffle(manager.data)

        model.train()
        start = time.perf_counter()
        train_loss = train_epoch(manager, criterion, optimizer, scalar, visualize)
        elapsed = timedelta(seconds=(time.perf_counter() - start))

        model.eval()
        with torch.no_grad():
            val_loss = train_epoch(manager, criterion, mode='eval')

        checkpoint = f'[{str(epoch + 1).rjust(len(str(manager.max_epochs)), "0")}]'
        checkpoint += f' Training PPL = {torch.exp(train_loss)}'
        checkpoint += f' | Validation PPL = {torch.exp(val_loss)}'
        checkpoint += f' | Elapsed Time = {elapsed}'
        logger.info(checkpoint); print()

        if val_loss < best_loss:
            i, best_loss = 0, val_loss
            manager.save_model()
        elif (i := i + 1) == manager.patience:
            break

    return score_model(manager, logger)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tqdm', action='store_true', help='enable tqdm')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--lang', nargs=2, metavar=('SRC', 'TGT'), required=True, help='language pair')
    parser.add_argument('--data', metavar='FILE', required=True, help='training data')
    parser.add_argument('--test', metavar='FILE', required=True, help='validation data')
    parser.add_argument('--vocab', metavar='FILE', required=True, help='shared vocab')
    parser.add_argument('--codes', metavar='FILE', required=True, help='shared codes')
    parser.add_argument('--model', metavar='FILE', required=True, help='save model')
    args, unknown = parser.parse_known_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    src_lang, tgt_lang = args.lang
    with open('config.toml') as config_file:
        config = toml.load(config_file)
    assert torch.cuda.is_available()
    device = torch.device('cuda')

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:], unknown[i + 1]
            config[option] = (int if value.isdigit() else float)(value)

    with open(args.vocab) as vocab_file, open(args.codes) as codes_file, \
            open(args.data) as data_file, open(args.test) as test_file:
        manager = Manager(src_lang, tgt_lang, vocab_file, codes_file,
            args.model, config, device, data_file, test_file)

    if torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')
    # if torch.__version__ >= '2.0':
    #     manager.model = torch.compile(manager.model)

    logger = logging.getLogger('torch.logger')
    logger.addHandler(logging.FileHandler(args.model + '.log'))

    train_model(manager, logger, args.tqdm)

if __name__ == '__main__':
    import argparse, logging
    main()
