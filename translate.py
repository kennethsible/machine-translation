import torch

from decoder import beam_search
from manager import Manager, Tokenizer


def translate_file(data_file: str, manager: Manager, tokenizer: Tokenizer) -> list[str]:
    with open(data_file) as file:
        return [translate_string(line, manager, tokenizer) for line in file]


def translate_string(string: str, manager: Manager, tokenizer: Tokenizer) -> str:
    model, vocab, device = manager.model, manager.vocab, manager.device
    src_words = ['<BOS>'] + tokenizer.tokenize(string).split() + ['<EOS>']

    model.eval()
    with torch.no_grad():
        src_nums, src_mask = torch.tensor(vocab.numberize(src_words)), None
        src_encs = model.encode(src_nums.unsqueeze(0).to(device), src_mask)
        out_nums = beam_search(manager, src_encs, src_mask, manager.beam_size)

    return tokenizer.detokenize(vocab.denumberize(out_nums.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='FILE', required=True, help='model file (.pt)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--string', metavar='STRING', help='input string')
    group.add_argument('--file', metavar='FILE', help='input file')
    args, unknown = parser.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dict = torch.load(args.model, map_location=device)
    src_lang, tgt_lang = model_dict['src_lang'], model_dict['tgt_lang']
    vocab_list, codes_list = model_dict['vocab_list'], model_dict['codes_list']

    config = model_dict['model_config']
    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:].replace('-', '_'), unknown[i + 1]
            config[option] = (int if value.isdigit() else float)(value)

    manager = Manager(
        src_lang,
        tgt_lang,
        config,
        device,
        args.model,
        vocab_list,
        codes_list,
        data_file=None,
        test_file=None,
    )
    manager.model.load_state_dict(model_dict['state_dict'])
    tokenizer = Tokenizer(manager.bpe, src_lang, tgt_lang)

    if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    if args.file:
        data_file = open(args.file)
        print(*translate_file(data_file, manager, tokenizer), sep='\n')
        data_file.close()
    elif args.string:
        print(translate_string(args.string, manager, tokenizer))


if __name__ == '__main__':
    import argparse

    main()
