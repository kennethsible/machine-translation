#!/bin/bash

echo "Downloading Europarl v7 Training Corpus..."
wget -nc -nv -O data/train.tgz \
  http://www.statmt.org/europarl/v7/de-en.tgz
tar -xvzf data/train.tgz -C data
mv data/europarl-v7.de-en.de data/train.de
mv data/europarl-v7.de-en.en data/train.en

# echo -e "\nDownloading WMT16 Development/Test Data..."
# wget -nc -nv -O data/dev.tgz \
#   http://data.statmt.org/wmt16/translation-task/dev.tgz
# wget -nc -nv -O data/test.tgz \
#   http://data.statmt.org/wmt16/translation-task/test.tgz

echo -e "\nPerforming Tokenization with Moses..."
sacremoses -l de -j 4 tokenize < data/train.de > data/train.tok.de
sacremoses -l en -j 4 tokenize < data/train.en > data/train.tok.en

echo -e "\nPerforming Tokenization with BPE..."
cat data/train.tok.de data/train.tok.en | subword-nmt learn-bpe -s 10000 -o data/bpe.out
subword-nmt apply-bpe -c data/bpe.out < data/train.tok.de > data/train.tok.bpe.de
subword-nmt apply-bpe -c data/bpe.out < data/train.tok.en > data/train.tok.bpe.en

echo -e "\nExtracting Shared Vocab with BPE..."
cat data/train.tok.bpe.de data/train.tok.bpe.en | subword-nmt get-vocab > data/vocab.bpe
wc -l data/vocab.bpe
python << END
with open('data/vocab.bpe') as vocab_file:
    vocab = {}
    for line in vocab_file.readlines():
        word, count = line.split(' ')
        vocab[word] = int(count)
    ratio = sum([1 if vocab[word] >= 100 else 0 for word in vocab]) / len(vocab)
    print(round(ratio * 100, 2))
END # https://aclanthology.org/2020.findings-emnlp.352/

echo -e "\nCombining Source and Target Data..."
paste data/train.tok.bpe.de data/train.tok.bpe.en > data/train.tok.bpe.de-en
paste data/train.tok.bpe.en data/train.tok.bpe.de > data/train.tok.bpe.en-de
wc -l data/train.tok.bpe.de-en

echo -e "\nDone."
