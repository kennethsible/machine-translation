#!/bin/bash

src_lang=$1
tgt_lang=$2

mkdir -p data data/train data/dev data/test

echo "Downloading Europarl v7 Training Corpus..."
wget -q -O data/train.tgz "http://www.statmt.org/europarl/v7/$src_lang-$tgt_lang.tgz" --show-progress
tar -xzf data/train.tgz -C data && rm data/train.tgz
mv "data/europarl-v7.$src_lang-$tgt_lang.$src_lang" "data/train/train.$src_lang"
mv "data/europarl-v7.$src_lang-$tgt_lang.$tgt_lang" "data/train/train.$tgt_lang"

echo -e "\nDownloading WMT 2016 Dev/Test Data..."
wget -q -O data/dev.tgz "http://data.statmt.org/wmt16/translation-task/dev.tgz" --show-progress
tar -xzf data/dev.tgz -C data && rm data/dev.tgz
find data/dev -type f ! -name "*$src_lang$tgt_lang*" -delete
mv "data/dev/newstest2014-$src_lang$tgt_lang-src.$src_lang.sgm" "data/dev/dev.$src_lang.sgm"
mv "data/dev/newstest2014-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "data/dev/dev.$tgt_lang.sgm"
mv "data/dev/newstest2015-$src_lang$tgt_lang-src.$src_lang.sgm" "data/test/test.$src_lang.sgm"
mv "data/dev/newstest2015-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "data/test/test.$tgt_lang.sgm"
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
sgm_to_txt "data/dev/dev.$src_lang.sgm" > "data/dev/dev.$src_lang"
sgm_to_txt "data/dev/dev.$tgt_lang.sgm" > "data/dev/dev.$tgt_lang"
sgm_to_txt "data/test/test.$src_lang.sgm" > "data/test/test.$src_lang"
sgm_to_txt "data/test/test.$tgt_lang.sgm" > "data/test/test.$tgt_lang"
rm data/dev/*.sgm data/test/*.sgm

echo -e "\nPerforming Tokenization with Moses..."
sacremoses -l $src_lang -j 4 tokenize < "data/train/train.$src_lang" > "data/train/train.tok.$src_lang"
sacremoses -l $tgt_lang -j 4 tokenize < "data/train/train.$tgt_lang" > "data/train/train.tok.$tgt_lang"
sacremoses -l $src_lang -j 4 tokenize < "data/dev/dev.$src_lang" > "data/dev/dev.tok.$src_lang"
sacremoses -l $tgt_lang -j 4 tokenize < "data/dev/dev.$tgt_lang" > "data/dev/dev.tok.$tgt_lang"
sacremoses -l $src_lang -j 4 tokenize < "data/test/test.$src_lang" > "data/test/test.tok.$src_lang"
sacremoses -l $tgt_lang -j 4 tokenize < "data/test/test.$tgt_lang" > "data/test/test.tok.$tgt_lang"

echo -e "\nPerforming Tokenization with BPE..."
cat "data/train/train.tok.$src_lang" "data/train/train.tok.$tgt_lang" | subword-nmt learn-bpe -s 32000 -o "data/codes.$src_lang$tgt_lang"
ln -s "codes.$src_lang$tgt_lang" "data/codes.$tgt_lang$src_lang"
subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "data/train/train.tok.$src_lang" > "data/train/train.tok.bpe.$src_lang"
subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "data/train/train.tok.$tgt_lang" > "data/train/train.tok.bpe.$tgt_lang"
subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "data/dev/dev.tok.$src_lang" > "data/dev/dev.tok.bpe.$src_lang"
subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "data/dev/dev.tok.$tgt_lang" > "data/dev/dev.tok.bpe.$tgt_lang"
subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "data/test/test.tok.$src_lang" > "data/test/test.tok.bpe.$src_lang"
subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "data/test/test.tok.$tgt_lang" > "data/test/test.tok.bpe.$tgt_lang"

echo -e "\nExtracting Shared Vocab with BPE..."
cat "data/train/train.tok.bpe.$src_lang" "data/train/train.tok.bpe.$tgt_lang" | subword-nmt get-vocab > "data/vocab.$src_lang$tgt_lang"
ln -s "vocab.$src_lang$tgt_lang" "data/vocab.$tgt_lang$src_lang"
wc -l "data/vocab.$src_lang$tgt_lang"

echo -e "\nCombining Source and Target Data..."
paste "data/train/train.tok.bpe.$src_lang" "data/train/train.tok.bpe.$tgt_lang" > "data/train/train.tok.bpe.$src_lang$tgt_lang"
paste "data/train/train.tok.bpe.$tgt_lang" "data/train/train.tok.bpe.$src_lang" > "data/train/train.tok.bpe.$tgt_lang$src_lang"
wc -l "data/train/train.tok.bpe.$src_lang$tgt_lang"
paste "data/dev/dev.tok.bpe.$src_lang" "data/dev/dev.tok.bpe.$tgt_lang" > "data/dev/dev.tok.bpe.$src_lang$tgt_lang"
paste "data/dev/dev.tok.bpe.$tgt_lang" "data/dev/dev.tok.bpe.$src_lang" > "data/dev/dev.tok.bpe.$tgt_lang$src_lang"
wc -l "data/dev/dev.tok.bpe.$src_lang$tgt_lang"
paste "data/test/test.tok.bpe.$src_lang" "data/test/test.tok.bpe.$tgt_lang" > "data/test/test.tok.bpe.$src_lang$tgt_lang"
paste "data/test/test.tok.bpe.$tgt_lang" "data/test/test.tok.bpe.$src_lang" > "data/test/test.tok.bpe.$tgt_lang$src_lang"
wc -l "data/test/test.tok.bpe.$src_lang$tgt_lang"

echo -e "\nCleaning Train/Dev/Test Data..."
awk -i inplace '!seen[$0]++' "data/train/train.tok.bpe.$src_lang$tgt_lang"
awk -i inplace '!seen[$0]++' "data/train/train.tok.bpe.$tgt_lang$src_lang"
wc -l "data/train/train.tok.bpe.$src_lang$tgt_lang"
awk -i inplace '!seen[$0]++' "data/dev/dev.tok.bpe.$src_lang$tgt_lang"
awk -i inplace '!seen[$0]++' "data/dev/dev.tok.bpe.$tgt_lang$src_lang"
wc -l "data/dev/dev.tok.bpe.$src_lang$tgt_lang"
awk -i inplace '!seen[$0]++' "data/test/test.tok.bpe.$src_lang$tgt_lang"
awk -i inplace '!seen[$0]++' "data/test/test.tok.bpe.$tgt_lang$src_lang"
wc -l "data/test/test.tok.bpe.$src_lang$tgt_lang"

echo -e "\nDone."
