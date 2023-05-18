from manager import Manager
from decoder import beam_search
import torch

def translate_file(data_file, manager):
    return [translate_string(line, manager) for line in data_file]

def translate_string(string, manager):
    model, vocab, device = manager.model, manager.vocab, manager.device
    src_words = ['<BOS>'] + manager.tokenize(string).split() + ['<EOS>']

    model.eval()
    with torch.no_grad():
        src_nums, src_mask = torch.tensor(vocab.numberize(src_words)), None
        src_encs = model.encode(src_nums.unsqueeze(0).to(device), src_mask)
        out_nums = beam_search(manager, src_encs, src_mask, manager.beam_size)

    return manager.detokenize(vocab.denumberize(out_nums)[1:-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='FILE', required=True, help='model file (.pt)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--string', metavar='STRING', help='input string')
    group.add_argument('--file', metavar='FILE', help='input file')
    args, unknown = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = torch.load(args.model, map_location=device)
    src_lang, tgt_lang = model_dict['src_lang'], model_dict['tgt_lang']
    vocab_file, codes_file = model_dict['vocab_file'], model_dict['codes_file']

    config = model_dict['config']
    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:].replace('-', '_'), unknown[i + 1]
            config[option] = (int if value.isdigit() else float)(value)

    manager = Manager(src_lang, tgt_lang, vocab_file, codes_file,
        args.model, config, device, data_file=None, test_file=None)
    manager.model.load_state_dict(model_dict['state_dict'])

    if torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')
    # if torch.__version__ >= '2.0':
    #     manager.model = torch.compile(manager.model)

    if args.file:
        data_file = open(args.file)
        print(*translate_file(data_file, manager), sep='\n')
        data_file.close()
    elif args.string:
        print(translate_string(args.string, manager))

if __name__ == '__main__':
    import argparse
    main()
