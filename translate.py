from sacremoses import MosesTokenizer, MosesDetokenizer
from decoding import greedy_search, beam_search
from subword_nmt.apply_bpe import BPE
from manager import Vocab
from model import Model
import torch, json, re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize(input, src_lang):
    return MosesTokenizer(src_lang).tokenize(input, return_str=True)

def detokenize(output, tgt_lang):
    output = MosesDetokenizer(tgt_lang).detokenize(output)
    return re.sub('(@@ )|(@@ ?$)', '', output)

def translate(input, vocab_file, codes_file, model_file, src_lang, tgt_lang, config):
    with open(codes_file) as file:
        input = BPE(file).process_line(tokenize(input, src_lang))
    words = ['<BOS>'] + input.split() + ['<EOS>']

    vocab = Vocab()
    with open(vocab_file) as file:
        for line in file:
            vocab.add(line.split()[0])

    model = Model(vocab.size()).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    with torch.no_grad():
        src_nums = vocab.numberize(*words).unsqueeze(0)
        memory = model.encode(src_nums, src_mask=None)
        model_out = greedy_search(model, memory) if config['beam_size'] is None \
            else beam_search(model, memory, config['beam_size'])
    return detokenize(vocab.denumberize(*model_out), tgt_lang)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    parser.add_argument('--vocab', metavar='FILE', help='vocab (from BPE)')
    parser.add_argument('--codes', metavar='FILE', help='codes (from BPE)')
    parser.add_argument('--config', metavar='FILE', default='config.json', help='model config')
    parser.add_argument('--load', metavar='FILE', help='load state_dict')
    parser.add_argument('input', metavar='STRING', help='input string')
    args = parser.parse_args()

    src_lang, tgt_lang = args.lang
    if not args.vocab:
        args.vocab = f'data/vocab.{src_lang}{tgt_lang}'
    if not args.codes:
        args.codes = f'data/codes.{src_lang}{tgt_lang}'
    if not args.load:
        args.load = f'model.{src_lang}{tgt_lang}'

    with open(args.config, 'r') as file:
        config = json.load(file)

    print(translate(args.input, args.vocab, args.codes, args.load, *args.lang, config))
