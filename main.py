from datetime import timedelta
from manager import Manager
from score import score_model
import logging, random, torch
import math, toml, time, tqdm

Criterion = torch.nn.CrossEntropyLoss
Optimizer = torch.optim.Optimizer
Scaler = torch.cuda.amp.GradScaler
Logger = logging.Logger

def train_epoch(manager: Manager, criterion: Criterion, optimizer: Optimizer | None = None,
        scaler: Scaler | None = None, use_tqdm: bool = False) -> float:
    data = manager.data if optimizer else manager.test

    total_loss, num_tokens = 0., 0
    for batch in tqdm.tqdm(data, disable=(not use_tqdm)):
        src_nums, src_mask = batch.src_nums, batch.src_mask
        tgt_nums, tgt_mask = batch.tgt_nums, batch.tgt_mask

        with torch.cuda.amp.autocast():
            logits = manager.model(src_nums, src_mask, tgt_nums[:, :-1], tgt_mask)
            loss = criterion(torch.flatten(logits, 0, 1), torch.flatten(tgt_nums[:, 1:]))

        if optimizer and scaler:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                manager.model.parameters(),
                manager.clip_grad)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        num_tokens += batch.length()
        del logits, loss

    return total_loss / num_tokens

def train_model(manager: Manager, logger: Logger, use_tqdm: bool = False) -> tuple[tuple, list[str]]:
    model, vocab = manager.model, manager.vocab
    assert manager.data and len(manager.data) > 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.PAD,
        label_smoothing=manager.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=manager.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
         factor=manager.decay_factor, patience=manager.patience)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = torch.inf
    for epoch in range(manager.max_epochs):
        random.shuffle(manager.data)

        model.train()
        start = time.perf_counter()
        train_loss = train_epoch(manager, criterion, optimizer, scaler, use_tqdm)
        elapsed = timedelta(seconds=(time.perf_counter() - start))

        model.eval()
        with torch.no_grad():
            val_loss = train_epoch(manager, criterion, use_tqdm=use_tqdm)
        scheduler.step(val_loss)

        checkpoint = f'[{str(epoch + 1).rjust(len(str(manager.max_epochs)), "0")}]'
        checkpoint += f' Training PPL = {math.exp(train_loss):.16f}'
        checkpoint += f' | Validation PPL = {math.exp(val_loss):.16f}'
        checkpoint += f' | Learning Rate = {optimizer.param_groups[0]["lr"]:.16f}'
        checkpoint += f' | Elapsed Time = {elapsed}'
        logger.info(checkpoint); print()

        if val_loss < best_loss:
            manager.save_model()
            best_loss = val_loss
        if optimizer.param_groups[0]['lr'] < manager.min_lr:
            break

    return score_model(manager, logger, use_tqdm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar=('SRC', 'TGT'), required=True, help='language pair')
    parser.add_argument('--data', metavar='FILE', required=True, help='training data')
    parser.add_argument('--test', metavar='FILE', required=True, help='testing data')
    parser.add_argument('--vocab', metavar='FILE', required=True, help='vocab file (shared)')
    parser.add_argument('--codes', metavar='FILE', required=True, help='codes file (shared)')
    parser.add_argument('--model', metavar='FILE', required=True, help='model file (.pt)')
    parser.add_argument('--config', metavar='FILE', required=True, help='config file (.toml)')
    parser.add_argument('--log', metavar='FILE', required=True, help='log file (.md)')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--tqdm', action='store_true', help='import tqdm')
    args, unknown = parser.parse_known_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    src_lang, tgt_lang = args.lang
    with open(args.config) as config_file:
        config = toml.load(config_file)
    assert torch.cuda.is_available()
    device = torch.device('cuda')

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:].replace('-', '_'), unknown[i + 1]
            config[option] = (int if value.isdigit() else float)(value)

    with open(args.vocab) as vocab_file, open(args.codes) as codes_file, \
            open(args.data) as data_file, open(args.test) as test_file:
        manager = Manager(src_lang, tgt_lang, vocab_file, codes_file,
            args.model, config, device, data_file, test_file)

    if torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    logger = logging.getLogger('torch.logger')
    logger.addHandler(logging.FileHandler(args.log))

    train_model(manager, logger, args.tqdm)

if __name__ == '__main__':
    import argparse
    main()
