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
for path in "data/dev/dev" "data/test/test"
do
    sgm_to_txt "$path.$src_lang.sgm" > "$path.$src_lang"
    sgm_to_txt "$path.$tgt_lang.sgm" > "$path.$tgt_lang"
done
rm data/dev/*.sgm data/test/*.sgm

echo -e "\nPerforming Tokenization with Moses..."
for path in "data/train/train" "data/dev/dev" "data/test/test"
do
    sacremoses -l $src_lang -j 4 tokenize < "$path.$src_lang" > "$path.tok.$src_lang"
    sacremoses -l $tgt_lang -j 4 tokenize < "$path.$tgt_lang" > "$path.tok.$tgt_lang"
done

echo -e "\nPerforming Tokenization with BPE..."
cat "data/train/train.tok.$src_lang" "data/train/train.tok.$tgt_lang" \
    | subword-nmt learn-bpe -s 32000 -o "data/codes.$src_lang$tgt_lang"
ln -s "codes.$src_lang$tgt_lang" "data/codes.$tgt_lang$src_lang"
for path in "data/train/train" "data/dev/dev" "data/test/test"
do
    subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "$path.tok.$src_lang" > "$path.tok.bpe.$src_lang"
    subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "$path.tok.$tgt_lang" > "$path.tok.bpe.$tgt_lang"
done

echo -e "\nExtracting Shared Vocab with BPE..."
cat "data/train/train.tok.bpe.$src_lang" "data/train/train.tok.bpe.$tgt_lang" \
    | subword-nmt get-vocab > "data/vocab.$src_lang$tgt_lang"
ln -s "vocab.$src_lang$tgt_lang" "data/vocab.$tgt_lang$src_lang"
wc -l "data/vocab.$src_lang$tgt_lang"

echo -e "\nCombining Source and Target Data..."
for path in "data/train/train" "data/dev/dev" "data/test/test"
do
    paste "$path.tok.bpe.$src_lang" "$path.tok.bpe.$tgt_lang" > "$path.tok.bpe.$src_lang$tgt_lang"
    paste "$path.tok.bpe.$tgt_lang" "$path.tok.bpe.$src_lang" > "$path.tok.bpe.$tgt_lang$src_lang"
    wc -l "$path.tok.bpe.$src_lang$tgt_lang"
done

echo -e "\nCleaning Train/Dev/Test Data..."
for path in "data/train/train" "data/dev/dev" "data/test/test"
do
    awk -i inplace '!seen[$0]++' "$path.tok.bpe.$src_lang$tgt_lang"
    awk -i inplace '!seen[$0]++' "$path.tok.bpe.$tgt_lang$src_lang"
    wc -l "$path.tok.bpe.$src_lang$tgt_lang"
done

echo -e "\nDone."
