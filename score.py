from model import Model
from manager import Vocab, Batch, load_data, device
from decode import beam_search
from translate import detokenize
from sacrebleu.metrics import BLEU, CHRF
from datetime import timedelta
import torch, time, json

bleu, chrf = BLEU(), CHRF()

def score_model(config, indent=0):
    if isinstance(config['vocab'], Vocab):
        vocab = config['vocab']
    else:
        vocab = Vocab()
        with open(config['vocab']) as file:
            for line in file:
                vocab.add(line.split()[0])
        assert vocab.size() > 0

    if isinstance(config['data'], list):
        test_data = []
        for batch in config['data']:
            for i in range(batch.size()):
                src_nums = batch._src_nums[i].unsqueeze(0)
                tgt_nums = batch._tgt_nums[i].unsqueeze(0)
                test_data.append(Batch(src_nums, tgt_nums, vocab.padding_idx))
    else:
        test_data = load_data(config['data'], vocab)
        assert len(test_data) > 0

    if isinstance(config['load'], Model):
        model = config['load']
    else:
        model = Model(vocab.size()).to(device)
        model.load_state_dict(torch.load(config['load'], map_location=device))
        model.eval()

    start = time.time()
    candidate, reference = [], []
    with torch.no_grad():
        for batch in test_data:
            src_encs = model.encode(batch.src_nums, batch.src_mask)
            out = beam_search(model, src_encs, config['beam_size'], batch.src_mask)
            reference.append(detokenize(vocab.denumberize(
                *[num for num in batch._tgt_nums[0, 1:-1] if num != vocab.padding_idx]
            ), config['lang']))
            candidate.append(detokenize(vocab.denumberize(*out), config['lang']))

    bleu_score = bleu.corpus_score(candidate, [reference])
    chrf_score = chrf.corpus_score(candidate, [reference])
    elapsed = timedelta(seconds=(time.time() - start))

    output = f'BLEU = {bleu_score.score} | chrF2 = {chrf_score.score} | Decode Time: {elapsed}'
    if config['out']:
        with open(config['out'], 'w') as file:
            file.write(output + '\n\n')
            for translation in candidate:
                file.write(translation + '\n')
    print(indent * ' ' + output)

    return bleu_score, chrf_score

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    parser.add_argument('--data', metavar='FILE', help='testing data')
    parser.add_argument('--vocab', metavar='FILE', help='shared vocab')
    parser.add_argument('--config', metavar='FILE', default='model.config', help='model config')
    parser.add_argument('--load', metavar='FILE', help='load state_dict')
    parser.add_argument('--out', metavar='FILE', help='save output')
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
        else f'data/testing/test.tok.bpe.{src_lang}{tgt_lang}'
    config['vocab'] = args.vocab if args.vocab \
        else f'data/vocab.{src_lang}{tgt_lang}'
    config['load'] = args.load if args.load \
        else f'data/output/model.{src_lang}{tgt_lang}'
    config['out'] = args.out if args.out \
        else f'data/output/out.{src_lang}{tgt_lang}'

    score_model(config)
