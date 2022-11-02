#!/bin/bash

merge_ops=$2
src_lang=$3
tgt_lang=$4

usage="usage: preprocess.sh MERGE_OPS SRC_LANG TGT_LANG"
if [ -z "$merge_ops" ] || [ -z "$src_lang" ] || [ -z "$tgt_lang" ]; then
    echo -e "missing positional argument(s)\n$usage"
    exit 1
fi

mkdir -p data data/training data/validation data/testing data/output

echo "Downloading WMT16 Training Corpora..."
for path in "data/training"; do
    touch "$path/train.$src_lang" "$path/train.$tgt_lang"
    wget -q -O data/europarl_v7.tgz "http://www.statmt.org/europarl/v7/$src_lang-$tgt_lang.tgz" --show-progress
    tar -xzf data/europarl_v7.tgz -C data && rm data/europarl_v7.tgz
    mv "data/europarl-v7.$src_lang-$tgt_lang.$src_lang" "$path/europarl_v7.$src_lang"
    mv "data/europarl-v7.$src_lang-$tgt_lang.$tgt_lang" "$path/europarl_v7.$tgt_lang"
    cat "$path/europarl_v7.$src_lang" >> "$path/train.$src_lang"
    cat "$path/europarl_v7.$tgt_lang" >> "$path/train.$tgt_lang"
    rm "$path/europarl_v7.$src_lang" "$path/europarl_v7.$tgt_lang"
done

echo -e "\nDownloading WMT16 Validation Data..."
wget -q -O data/dev.tgz "http://data.statmt.org/wmt16/translation-task/dev.tgz" --show-progress
tar -xzf data/dev.tgz -C data && rm data/dev.tgz
find data/dev -type f ! -name "*$src_lang$tgt_lang*" -delete
mv "data/dev/newstest2014-$src_lang$tgt_lang-src.$src_lang.sgm" "data/validation/val.$src_lang.sgm"
mv "data/dev/newstest2014-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "data/validation/val.$tgt_lang.sgm"
mv "data/dev/newstest2015-$src_lang$tgt_lang-src.$src_lang.sgm" "data/testing/test.$src_lang.sgm"
mv "data/dev/newstest2015-$src_lang$tgt_lang-ref.$tgt_lang.sgm" "data/testing/test.$tgt_lang.sgm"
rm -r data/dev
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
for path in "data/validation/val" "data/testing/test"; do
    sgm_to_txt "$path.$src_lang.sgm" > "$path.$src_lang"
    sgm_to_txt "$path.$tgt_lang.sgm" > "$path.$tgt_lang"
done
rm data/validation/*.sgm data/testing/*.sgm

echo -e "\nPerforming Tokenization with Moses..."
for path in "data/training/train" "data/validation/val" "data/testing/test"; do
    sacremoses -l $src_lang -j 4 tokenize < "$path.$src_lang" > "$path.tok.$src_lang"
    sacremoses -l $tgt_lang -j 4 tokenize < "$path.$tgt_lang" > "$path.tok.$tgt_lang"
done

echo -e "\nLearning BPE for Subword Tokenization..."
cat "data/training/train.tok.$src_lang" "data/training/train.tok.$tgt_lang" \
    | subword-nmt learn-bpe -s $merge_ops -o "data/codes.$src_lang$tgt_lang"
ln -s "codes.$src_lang$tgt_lang" "data/codes.$tgt_lang$src_lang"

echo -e "\nExtracting Shared Vocab with BPE..."
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

echo -e "\nCleaning WMT16 Parallel Data..."
for path in "data/training/train" "data/validation/val" "data/testing/test"; do
    awk -i inplace '!seen[$0]++' "$path.tok.bpe.$src_lang$tgt_lang"
    awk -i inplace '!seen[$0]++' "$path.tok.bpe.$tgt_lang$src_lang"
    wc -l "$path.tok.bpe.$src_lang$tgt_lang"
done

echo -e "\nDone."
