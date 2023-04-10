#!/bin/bash

commoncrawl=0
europarl_v7=0
src_lang=$2
tgt_lang=$3
merge_ops=$4

while getopts ":ce" opt; do
   case $opt in
      c) commoncrawl=1;;
      e) europarl_v7=1;;
   esac
done

usage="usage: preprocess.sh [-ce] SRC_LANG TGT_LANG MERGE_OPS"
if [ -z "$src_lang" ] || [ -z "$tgt_lang" ] || [ -z "$merge_ops" ]; then
    echo -e "$usage"
    exit 1
fi

mkdir -p data data/training

echo "[1/10] Downloading WMT17 Training Data..."
if [ $commoncrawl -eq 0 ] && [ $europarl_v7 -eq 0 ]; then
    echo -e "$usage"
    exit 1
fi
for path in "data/training"; do
    if [ $commoncrawl -eq 1 ]; then
        wget -q -O data/commoncrawl.tgz "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz" --show-progress
        tar -xzf data/commoncrawl.tgz -C data && rm data/commoncrawl.tgz
        cat "data/commoncrawl.$src_lang-$tgt_lang.$src_lang" >> "$path/data.$src_lang"
        cat "data/commoncrawl.$src_lang-$tgt_lang.$tgt_lang" >> "$path/data.$tgt_lang"
        wc -l "$path/data.$src_lang"
    fi

    if [ $europarl_v7 -eq 1 ]; then
        wget -q -O data/europarl-v7.tgz "http://www.statmt.org/europarl/v7/$src_lang-$tgt_lang.tgz" --show-progress
        tar -xzf data/europarl-v7.tgz -C data && rm data/europarl-v7.tgz
        cat "data/europarl-v7.$src_lang-$tgt_lang.$src_lang" >> "$path/data.$src_lang"
        cat "data/europarl-v7.$src_lang-$tgt_lang.$tgt_lang" >> "$path/data.$tgt_lang"
        wc -l "$path/data.$src_lang"
    fi 
done
find data -maxdepth 1 -type f -delete

echo -e "\n[2/10] Tokenizing Training Data..."
for path in "data/training"; do
    sacremoses -l $src_lang -j 4 tokenize < "$path/data.$src_lang" > "$path/data.tok.$src_lang"
    sacremoses -l $tgt_lang -j 4 tokenize < "$path/data.$tgt_lang" > "$path/data.tok.$tgt_lang"
done

function sgm2txt {
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

echo -e "\n[3/10] Downloading WMT17 Validation Data..."
wget -q -O data/dev.tgz "http://data.statmt.org/wmt17/translation-task/dev.tgz" --show-progress
tar -xzf data/dev.tgz -C data && rm data/dev.tgz
mv data/dev data/validation
for path in "data/validation"; do
    mv "$path/newstest2016-$src_lang$tgt_lang-src.$src_lang.sgm" "$path/data.$src_lang.sgm"
    mv "$path/newstest2016-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "$path/data.$tgt_lang.sgm"  
    sgm2txt "$path/data.$src_lang.sgm" > "$path/data.$src_lang"
    sgm2txt "$path/data.$tgt_lang.sgm" > "$path/data.$tgt_lang"
done
find data/validation -type f ! -name "*data.$src_lang" -and ! -name "*data.$tgt_lang" -delete

echo -e "\n[4/10] Tokenizing Validation Data..."
for path in "data/validation"; do
    sacremoses -l $src_lang -j 4 tokenize < "$path/data.$src_lang" > "$path/data.tok.$src_lang"
    sacremoses -l $tgt_lang -j 4 tokenize < "$path/data.$tgt_lang" > "$path/data.tok.$tgt_lang"
done

echo -e "\n[5/10] Downloading WMT17 Testing Data..."
wget -q -O data/test.tgz "http://data.statmt.org/wmt17/translation-task/test.tgz" --show-progress
tar -xzf data/test.tgz -C data && rm data/test.tgz
mv data/test data/testing
for path in "data/testing"; do
    mv "$path/newstest2017-$src_lang$tgt_lang-src.$src_lang.sgm" "$path/data.$src_lang.sgm"
    mv "$path/newstest2017-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "$path/data.$tgt_lang.sgm"  
    sgm2txt "$path/data.$src_lang.sgm" > "$path/data.$src_lang"
    sgm2txt "$path/data.$tgt_lang.sgm" > "$path/data.$tgt_lang"
done
find data/testing -type f ! -name "*data.$src_lang" -and ! -name "*data.$tgt_lang" -delete

echo -e "\n[6/10] Tokenizing Testing Data..."
for path in "data/testing"; do
    sacremoses -l $src_lang -j 4 tokenize < "$path/data.$src_lang" > "$path/data.tok.$src_lang"
    sacremoses -l $tgt_lang -j 4 tokenize < "$path/data.$tgt_lang" > "$path/data.tok.$tgt_lang"
done

echo -e "\n[7/10] Learning BPE for Subword Tokenization..."
cat "data/training/data.tok.$src_lang" "data/training/data.tok.$tgt_lang" \
    | subword-nmt learn-bpe -s $merge_ops -o "data/codes.$src_lang$tgt_lang"
ln -s "codes.$src_lang$tgt_lang" "data/codes.$tgt_lang$src_lang"

echo -e "\n[8/10] Applying BPE Subword Tokenization..."
for path in "data/training" "data/validation" "data/testing"; do
    subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "$path/data.tok.$src_lang" > "$path/data.tok.bpe.$src_lang"
    subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "$path/data.tok.$tgt_lang" > "$path/data.tok.bpe.$tgt_lang"
done
cat "data/training/data.tok.bpe.$src_lang" "data/training/data.tok.bpe.$tgt_lang" \
    | subword-nmt get-vocab > "data/vocab.$src_lang$tgt_lang"
ln -s "vocab.$src_lang$tgt_lang" "data/vocab.$tgt_lang$src_lang"
wc -l "data/vocab.$src_lang$tgt_lang"

echo -e "\n[9/10] Combining Source and Target Data..."
for path in "data/training" "data/validation" "data/testing"; do
    paste "$path/data.tok.bpe.$src_lang" "$path/data.tok.bpe.$tgt_lang" > "$path/data.tok.bpe.$src_lang$tgt_lang"
    paste "$path/data.tok.bpe.$tgt_lang" "$path/data.tok.bpe.$src_lang" > "$path/data.tok.bpe.$tgt_lang$src_lang"
    wc -l "$path/data.tok.bpe.$src_lang$tgt_lang"
done

echo -e "\n[10/10] Cleaning Parallel Data..."
for path in "data/training" "data/validation" "data/testing"; do
    awk -i inplace '!seen[$0]++' "$path/data.tok.bpe.$src_lang$tgt_lang"
    awk -i inplace '!seen[$0]++' "$path/data.tok.bpe.$tgt_lang$src_lang"
    wc -l "$path/data.tok.bpe.$src_lang$tgt_lang"
done

echo -e "\nDone."
