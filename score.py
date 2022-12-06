from sacrebleu.metrics import BLEU, CHRF
from manager import Manager, Tokenizer
from decode import beam_search
from datetime import timedelta
import torch, time, toml

bleu, chrf = BLEU(), CHRF()

def score_model(manager, tokenizer, model_file=None, *, indent=0):
    start = time.perf_counter()
    manager.model.eval()

    candidate, reference = [], []
    with torch.no_grad():
        for batch in manager.test:
            src_nums, tgt_nums = batch.src_nums, batch.tgt_nums
            src_mask = batch.src_mask
            src_encs = manager.model.encode(src_nums, src_mask)

            for i in range(src_encs.size(0)):
                out_nums = beam_search(manager, src_encs[i], src_mask[i], manager.config['beam-width'])
                reference.append(tokenizer.detokenize(manager.vocab.denumberize(*tgt_nums[i])))
                candidate.append(tokenizer.detokenize(manager.vocab.denumberize(*out_nums)))

    bleu_score = bleu.corpus_score(candidate, [reference])
    chrf_score = chrf.corpus_score(candidate, [reference])
    elapsed = timedelta(seconds=(time.perf_counter() - start))

    status_update = indent * ' ' + f'BLEU = {bleu_score.score} | chrF2 = {chrf_score.score} | Decode Time: {elapsed}'
    if model_file:
        with open(model_file + '.log', 'a') as file:
            file.write(status_update + '\n')
    print(status_update)

    return bleu_score, chrf_score, candidate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    parser.add_argument('--data', metavar='FILE', help='testing data')
    parser.add_argument('--vocab', metavar='FILE', help='shared vocab')
    parser.add_argument('--config', metavar='FILE', default='model.config', help='model config')
    parser.add_argument('--load', metavar='FILE', help='load state_dict')
    args, unknown = parser.parse_known_args()

    src_lang, tgt_lang = args.lang
    with open(args.config) as file:
        config = toml.load(file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and arg[2:] in config:
            if len(unknown) >= i + 1:
                config[arg[2:]] = int(unknown[i + 1])

    if not args.data:
        args.data = f'data/testing/test.tok.bpe.{src_lang}{tgt_lang}'
    if not args.vocab:
        args.vocab = f'data/vocab.{src_lang}{tgt_lang}'
    if not args.load:
        args.load = f'data/model.{src_lang}{tgt_lang}'
        open(args.load + '.log', 'w').close()

    manager = Manager(
        src_lang,
        tgt_lang,
        config,
        device,
        None,
        args.data,
        args.vocab
    )
    if args.load:
        manager.load_model(args.load)
    tokenizer = Tokenizer(src_lang, tgt_lang)

    _, _, candidate = score_model(manager, tokenizer, args.load)
    print('', *candidate, sep='\n')

if __name__ == '__main__':
    import argparse
    main()
