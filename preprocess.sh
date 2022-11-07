#!/bin/bash

commentary_v12=0
commoncrawl=0
europarl_v7=0
src_lang=$2
tgt_lang=$3
merge_ops=$4

while getopts ":nce" opt; do
   case $opt in
      n) commentary_v12=1;;
      c) commoncrawl=1;;
      e) europarl_v7=1;;
   esac
done

usage="usage: preprocess.sh [-nce] SRC_LANG TGT_LANG MERGE_OPS"
if [ -z "$src_lang" ] || [ -z "$tgt_lang" ] || [ -z "$merge_ops" ]; then
    echo -e "$usage"
    exit 1
fi

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

mkdir -p data data/output data/training

echo "Downloading WMT17 Training Corpora..."
if [ $commentary_v12 -eq 0 ] && [ $commoncrawl -eq 0 ] && [ $europarl_v7 -eq 0 ]; then
    echo -e "$usage"
    exit 1
fi
for path in "data/training"; do
    if [ $commentary_v12 -eq 1 ]; then
        wget -q -O data/news-commentary-v12.tgz "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz" --show-progress
        tar -xzf data/news-commentary-v12.tgz -C data && rm data/news-commentary-v12.tgz
        find data/training -type f ! -name "*$src_lang-$tgt_lang*" -delete
        cat "$path/news-commentary-v12.$src_lang-$tgt_lang.$src_lang" >> "$path/train.$src_lang"
        cat "$path/news-commentary-v12.$src_lang-$tgt_lang.$tgt_lang" >> "$path/train.$tgt_lang"
        find data/training -type f -name "*$src_lang-$tgt_lang*" -delete
        wc -l "$path/train.$src_lang"
    fi

    if [ $commoncrawl -eq 1 ]; then
        wget -q -O data/commoncrawl.tgz "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz" --show-progress
        tar -xzf data/commoncrawl.tgz -C data && rm data/commoncrawl.tgz
        cat "data/commoncrawl.$src_lang-$tgt_lang.$src_lang" >> "$path/train.$src_lang"
        cat "data/commoncrawl.$src_lang-$tgt_lang.$tgt_lang" >> "$path/train.$tgt_lang"
        wc -l "$path/train.$src_lang"
    fi

    if [ $europarl_v7 -eq 1 ]; then
        wget -q -O data/europarl-v7.tgz "http://www.statmt.org/europarl/v7/$src_lang-$tgt_lang.tgz" --show-progress
        tar -xzf data/europarl-v7.tgz -C data && rm data/europarl-v7.tgz
        cat "data/europarl-v7.$src_lang-$tgt_lang.$src_lang" >> "$path/train.$src_lang"
        cat "data/europarl-v7.$src_lang-$tgt_lang.$tgt_lang" >> "$path/train.$tgt_lang"
        wc -l "$path/train.$src_lang"
    fi

    find data -maxdepth 1 -type f -delete
done

echo -e "\nDownloading WMT17 Validation Data..."
wget -q -O data/dev.tgz "http://data.statmt.org/wmt17/translation-task/dev.tgz" --show-progress
tar -xzf data/dev.tgz -C data && rm data/dev.tgz
mv data/dev data/validation
for path in "data/validation"; do
    mv "$path/newstest2016-$src_lang$tgt_lang-src.$src_lang.sgm" "$path/val.$src_lang.sgm"
    mv "$path/newstest2016-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "$path/val.$tgt_lang.sgm"  
    sgm_to_txt "$path/val.$src_lang.sgm" > "$path/val.$src_lang"
    sgm_to_txt "$path/val.$tgt_lang.sgm" > "$path/val.$tgt_lang"
done
find data/validation -type f ! -name "*val.$src_lang" -and ! -name "*val.$tgt_lang" -delete

echo -e "\nDownloading WMT17 Testing Data..."
wget -q -O data/test.tgz "http://data.statmt.org/wmt17/translation-task/test.tgz" --show-progress
tar -xzf data/test.tgz -C data && rm data/test.tgz
mv data/test data/testing
for path in "data/testing"; do
    mv "$path/newstest2017-$src_lang$tgt_lang-src.$src_lang.sgm" "$path/test.$src_lang.sgm"
    mv "$path/newstest2017-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "$path/test.$tgt_lang.sgm"  
    sgm_to_txt "$path/test.$src_lang.sgm" > "$path/test.$src_lang"
    sgm_to_txt "$path/test.$tgt_lang.sgm" > "$path/test.$tgt_lang"
done
find data/testing -type f ! -name "*test.$src_lang" -and ! -name "*test.$tgt_lang" -delete

echo -e "\nPerforming Tokenization with Moses..."
for path in "data/training/train" "data/validation/val" "data/testing/test"; do
    sacremoses -l $src_lang -j 4 tokenize < "$path.$src_lang" > "$path.tok.$src_lang"
    sacremoses -l $tgt_lang -j 4 tokenize < "$path.$tgt_lang" > "$path.tok.$tgt_lang"
done

echo -e "\nLearning BPE for Subword Tokenization..."
cat "data/training/train.tok.$src_lang" "data/training/train.tok.$tgt_lang" \
    | subword-nmt learn-bpe -s $merge_ops -o "data/codes.$src_lang$tgt_lang"
ln -s "codes.$src_lang$tgt_lang" "data/codes.$tgt_lang$src_lang"

echo -e "\nPerforming Subword Tokenization with BPE..."
for path in "data/training/train" "data/validation/val" "data/testing/test"; do
    subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "$path.tok.$src_lang" > "$path.tok.bpe.$src_lang"
    subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "$path.tok.$tgt_lang" > "$path.tok.bpe.$tgt_lang"
done
cat "data/training/train.tok.bpe.$src_lang" "data/training/train.tok.bpe.$tgt_lang" \
    | subword-nmt get-vocab > "data/vocab.$src_lang$tgt_lang"
ln -s "vocab.$src_lang$tgt_lang" "data/vocab.$tgt_lang$src_lang"
wc -l "data/vocab.$src_lang$tgt_lang"

echo -e "\nCombining Source and Target Data..."
for path in "data/training/train" "data/validation/val" "data/testing/test"; do
    paste "$path.tok.bpe.$src_lang" "$path.tok.bpe.$tgt_lang" > "$path.tok.bpe.$src_lang$tgt_lang"
    paste "$path.tok.bpe.$tgt_lang" "$path.tok.bpe.$src_lang" > "$path.tok.bpe.$tgt_lang$src_lang"
    wc -l "$path.tok.bpe.$src_lang$tgt_lang"
done

echo -e "\nCleaning WMT17 Parallel Data..."
for path in "data/training/train" "data/validation/val" "data/testing/test"; do
    awk -i inplace '!seen[$0]++' "$path.tok.bpe.$src_lang$tgt_lang"
    awk -i inplace '!seen[$0]++' "$path.tok.bpe.$tgt_lang$src_lang"
    wc -l "$path.tok.bpe.$src_lang$tgt_lang"
done

echo -e "\nDone."
