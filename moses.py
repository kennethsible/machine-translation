from sacremoses import MosesTokenizer
import argparse, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('infile', nargs='?', type=str, help='input data to tokenize')
parser.add_argument('-lang', '--lang', type=str, help='language of the input data')
parser.add_argument('-o', '--outfile', type=str, help='write tokenization to file')
args = parser.parse_args()

mt = MosesTokenizer(lang=args.lang)

with open(args.infile) as infile:
    data = infile.readlines()

with open(args.outfile, 'w') as outfile:
    for line in tqdm.tqdm(data):
        outfile.write(' '.join(mt.tokenize(line)) + '\n')
