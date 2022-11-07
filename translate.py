from sacremoses import MosesTokenizer, MosesDetokenizer
from subword_nmt.apply_bpe import BPE
from manager import Vocab, device
from model import Model
from decode import beam_search
import torch, toml, re

def tokenize(input, codes, src_lang):
    string = MosesTokenizer(src_lang).tokenize(input, return_str=True)
    return BPE(codes).process_line(string)

def detokenize(output, tgt_lang):
    string = MosesDetokenizer(tgt_lang).detokenize(output)
    return re.sub('(@@ )|(@@ ?$)', '', string)

def translate(input, config):
    vocab = Vocab()
    with open(config['vocab']) as file:
        for line in file:
            vocab.add(line.split()[0])
    assert vocab.size() > 0

    src_lang, tgt_lang = config['lang']
    with open(config['codes']) as file:
        input = tokenize(input, file, src_lang)
    assert len(input) > 0

    model = Model(vocab.size()).to(device)
    model.load_state_dict(torch.load(config['load'], map_location=device))
    model.eval()

    with torch.no_grad():
        src_nums = vocab.numberize(*input.split()).unsqueeze(0)
        src_encs = model.encode(src_nums.to(device), None)
        out_nums = beam_search(model, vocab, src_encs, None,
            config['max_length'], config['beam_size'])

    return detokenize(vocab.denumberize(*out_nums), tgt_lang)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    parser.add_argument('--vocab', metavar='FILE', help='shared vocab')
    parser.add_argument('--codes', metavar='FILE', help='shared codes')
    parser.add_argument('--config', metavar='FILE', default='model.config', help='model config')
    parser.add_argument('--load', metavar='FILE', help='load state_dict')
    parser.add_argument('input', metavar='STRING', help='input string')
    args, unknown = parser.parse_known_args()

    with open(args.config) as file:
        config = toml.load(file)

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and arg[2:] in config:
            if len(unknown) >= i + 1:
                config[arg[2:]] = int(unknown[i + 1])

    src_lang, tgt_lang = args.lang
    config['lang'] = args.lang

    config['vocab'] = args.vocab if args.vocab \
        else f'data/vocab.{src_lang}{tgt_lang}'
    config['codes'] = args.codes if args.codes \
        else f'data/codes.{src_lang}{tgt_lang}'
    config['load'] = args.load if args.load \
        else f'data/output/model.{src_lang}{tgt_lang}'

    print(translate(args.input, config))

if __name__ == '__main__':
    import argparse
    main()
