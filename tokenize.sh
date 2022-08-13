#!/bin/bash

src_lang=$1
tgt_lang=$2

mkdir -p data data/train

echo "Downloading Europarl v7 Training Corpus..."
wget -q -O data/train.tgz "http://www.statmt.org/europarl/v7/$src_lang-$tgt_lang.tgz" --show-progress
tar -xvzf data/train.tgz -C data && rm data/train.tgz
mv "data/europarl-v7.$src_lang-$tgt_lang.$src_lang" "data/train.$src_lang"
mv "data/europarl-v7.$src_lang-$tgt_lang.$tgt_lang" "data/train.$tgt_lang"

echo -e "\nDownloading WMT16 Test Set..."
wget -q -O data/test.tgz "http://data.statmt.org/wmt16/translation-task/test.tgz" --show-progress
tar -xvzf data/test.tgz -C data && rm data/test.tgz
find data/test -type f ! -name "*$src_lang$tgt_lang*" -delete
mv "data/test/newstest2016-$src_lang$tgt_lang-src.$src_lang.sgm" "data/test/test.$src_lang.sgm"
mv "data/test/newstest2016-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "data/test/test.$tgt_lang.sgm"
function sgm_to_txt {
    output="$(python - << END
import re
with open('$1') as infile:
    for line in infile.readlines():
        line = re.split(r'(<[^>]+>)', line.strip())[1:-1]
        if len(line) != 3: continue
        tag, sentence, _ = line
        if tag[1:-1].split(' ')[0] == 'seg':
            print(sentence)
END
)"
    echo "$output\n"
}
sgm_to_txt "data/test/test.$src_lang.sgm" > "data/test/test.$src_lang"
sgm_to_txt "data/test/test.$tgt_lang.sgm" > "data/test/test.$tgt_lang"
rm "data/test/test.$src_lang.sgm" "data/test/test.$tgt_lang.sgm"

echo -e "\nPerforming Tokenization with Moses..."
sacremoses -l $src_lang -j 4 tokenize < "data/train.$src_lang" > "data/train.tok.$src_lang"
sacremoses -l $tgt_lang -j 4 tokenize < "data/train.$tgt_lang" > "data/train.tok.$tgt_lang"
sacremoses -l $src_lang -j 4 tokenize < "data/test/test.$src_lang" > "data/test/test.tok.$src_lang"
sacremoses -l $tgt_lang -j 4 tokenize < "data/test/test.$tgt_lang" > "data/test/test.tok.$tgt_lang"

echo -e "\nPerforming Tokenization with BPE..."
cat "data/train.tok.$src_lang" "data/train.tok.$tgt_lang" | subword-nmt learn-bpe -s 32000 -o "data/bpe.$src_lang$tgt_lang"
ln -s "bpe.$src_lang$tgt_lang" "data/bpe.$tgt_lang$src_lang"
subword-nmt apply-bpe -c "data/bpe.$src_lang$tgt_lang" < "data/train.tok.$src_lang" > "data/train.tok.bpe.$src_lang"
subword-nmt apply-bpe -c "data/bpe.$src_lang$tgt_lang" < "data/train.tok.$tgt_lang" > "data/train.tok.bpe.$tgt_lang"
subword-nmt apply-bpe -c "data/bpe.$src_lang$tgt_lang" < "data/test/test.tok.$src_lang" > "data/test/test.tok.bpe.$src_lang"
subword-nmt apply-bpe -c "data/bpe.$src_lang$tgt_lang" < "data/test/test.tok.$tgt_lang" > "data/test/test.tok.bpe.$tgt_lang"

echo -e "\nExtracting Shared Vocab with BPE..."
cat "data/train.tok.bpe.$src_lang" "data/train.tok.bpe.$tgt_lang" | subword-nmt get-vocab > "data/vocab.$src_lang$tgt_lang"
ln -s "vocab.$src_lang$tgt_lang" "data/vocab.$tgt_lang$src_lang"
wc -l "data/vocab.$src_lang$tgt_lang"

echo -e "\nCombining Source and Target Data..."
paste "data/train.tok.bpe.$src_lang" "data/train.tok.bpe.$tgt_lang" > "data/train.tok.bpe.$src_lang$tgt_lang"
paste "data/train.tok.bpe.$tgt_lang" "data/train.tok.bpe.$src_lang" > "data/train.tok.bpe.$tgt_lang$src_lang"
mv data/train.* data/train
wc -l "data/train/train.tok.bpe.$src_lang$tgt_lang"
paste "data/test/test.tok.bpe.$src_lang" "data/test/test.tok.bpe.$tgt_lang" > "data/test/test.tok.bpe.$src_lang$tgt_lang"
paste "data/test/test.tok.bpe.$tgt_lang" "data/test/test.tok.bpe.$src_lang" > "data/test/test.tok.bpe.$tgt_lang$src_lang"
wc -l "data/test/test.tok.bpe.$src_lang$tgt_lang"

echo -e "\nDone."
