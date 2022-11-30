from manager import Tokenizer, Manager
from decode import beam_search
import torch, toml

def translate_file(infile, manager, tokenizer, model_file=None):
    if model_file:
        manager.load_model(model_file)
        manager.model.eval()

    with open(infile) as file:
        return [translate_string(line, manager, tokenizer) for line in file]

def translate_string(string, manager, tokenizer, model_file=None):
    if model_file:
        manager.load_model(model_file)
        manager.model.eval()

    string = tokenizer.tokenize(string)
    assert len(string) > 0

    with torch.no_grad():
        max_length = manager.config['max-length']
        if max_length: max_length -= 2

        src_words = string.split()
        if max_length and len(src_words) > max_length:
            src_words = src_words[:max_length]
        src_words = ['<BOS>'] + src_words + ['<EOS>']

        src_nums = manager.vocab.numberize(*src_words).unsqueeze(0)
        src_encs = manager.model.encode(src_nums.to(manager.device), None)
        out_nums = beam_search(manager, src_encs, None, manager.config['beam-width'])

    return tokenizer.detokenize(manager.vocab.denumberize(*out_nums))

def interactive(manager, tokenizer, model_file=None):
    if model_file:
        manager.load_model(model_file)
        manager.model.eval()

    print('Interactive Mode (Ctrl+C to Quit)')
    try:
        while True:
            print(translate_string(input('\n> '), manager, tokenizer))
    except KeyboardInterrupt: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar='LANG', required=True, help='source/target language')
    parser.add_argument('--vocab', metavar='FILE', help='shared vocab')
    parser.add_argument('--codes', metavar='FILE', help='shared codes')
    parser.add_argument('--config', metavar='FILE', default='model.config', help='model config')
    parser.add_argument('--load', metavar='FILE', help='load state_dict')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', metavar='FILE', help='input file')
    group.add_argument('--string', metavar='STRING', help='input string')
    group.add_argument('--interactive', action='store_true', help='interactive session')
    args, unknown = parser.parse_known_args()

    src_lang, tgt_lang = args.lang
    with open(args.config) as file:
        config = toml.load(file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and arg[2:] in config:
            if len(unknown) >= i + 1:
                config[arg[2:]] = int(unknown[i + 1])

    if not args.vocab:
        args.vocab = f'data/vocab.{src_lang}{tgt_lang}'
    if not args.codes:
        args.codes = f'data/codes.{src_lang}{tgt_lang}'
    if not args.load:
        args.load = f'data/model.{src_lang}{tgt_lang}'

    manager = Manager(
        src_lang,
        tgt_lang,
        config,
        device,
        None,
        None,
        args.vocab
    )
    tokenizer = Tokenizer(src_lang, tgt_lang, args.codes)

    if args.file:
        print(*translate_file(args.file, manager, tokenizer, args.load), sep='\n')
    if args.string:
        print(translate_string(args.string, manager, tokenizer, args.load))
    elif args.interactive:
        interactive(manager, tokenizer, args.load)

if __name__ == '__main__':
    import argparse
    main()
