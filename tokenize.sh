#!/bin/bash

src_lang=$1
tgt_lang=$2

echo "Downloading Europarl v7 Training Corpus..."
wget -nc -nv -O data/train.tgz \
  "http://www.statmt.org/europarl/v7/$src_lang-$tgt_lang.tgz"
tar -xvzf data/train.tgz -C data
mv "data/europarl-v7.$src_lang-$tgt_lang.$src_lang" "data/train.$src_lang"
mv "data/europarl-v7.$src_lang-$tgt_lang.$tgt_lang" "data/train.$tgt_lang"

# echo -e "\nDownloading WMT16 Development/Test Data..."
# wget -nc -nv -O data/dev.tgz \
#   http://data.statmt.org/wmt16/translation-task/dev.tgz
# wget -nc -nv -O data/test.tgz \
#   http://data.statmt.org/wmt16/translation-task/test.tgz

echo -e "\nPerforming Tokenization with Moses..."
sacremoses -l $src_lang -j 4 tokenize < "data/train.$src_lang" > "data/train.tok.$src_lang"
sacremoses -l $tgt_lang -j 4 tokenize < "data/train.$tgt_lang" > "data/train.tok.$tgt_lang"

echo -e "\nPerforming Tokenization with BPE..."
cat "data/train.tok.$src_lang" "data/train.tok.$tgt_lang" | subword-nmt learn-bpe -s 10000 -o data/bpe.out
subword-nmt apply-bpe -c data/bpe.out < "data/train.tok.$src_lang" > "data/train.tok.bpe.$src_lang"
subword-nmt apply-bpe -c data/bpe.out < "data/train.tok.$tgt_lang" > "data/train.tok.bpe.$tgt_lang"
# https://aclanthology.org/2020.findings-emnlp.352/

echo -e "\nExtracting Shared Vocab with BPE..."
cat "data/train.tok.bpe.$src_lang" "data/train.tok.bpe.$tgt_lang" | subword-nmt get-vocab > data/vocab.bpe
wc -l data/vocab.bpe

echo -e "\nCombining Source and Target Data..."
paste "data/train.tok.bpe.$src_lang" "data/train.tok.bpe.$tgt_lang" > "data/train.tok.bpe.$src_lang-$tgt_lang"
paste "data/train.tok.bpe.$tgt_lang" "data/train.tok.bpe.$src_lang" > "data/train.tok.bpe.$tgt_lang-$src_lang"
wc -l "data/train.tok.bpe.$src_lang-$tgt_lang"

echo -e "\nDone."
