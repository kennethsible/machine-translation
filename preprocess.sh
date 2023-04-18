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
cp "data/codes.$src_lang$tgt_lang" "data/codes.$tgt_lang$src_lang"

function build_vocab {
    output="$(python - << END
src_data = set()
with open('$3/data.tok.bpe.$1') as src_file:
    for line in src_file.readlines():
        src_data.update(line.split())

tgt_data = set()
with open('$3/data.tok.bpe.$2') as tgt_file:
    for line in tgt_file.readlines():
        tgt_data.update(line.split())

# <UNK> SRC-TGT <EOS> TGT-ONLY <BOS> SRC-ONLY <PAD>
src_tgt, src_only, tgt_only = ['<UNK>\n'], [], []
with open('data/vocab.$1$2') as vocab_file:
    for line in vocab_file.readlines():
        word = line.split()[0]
        in_src, in_tgt = word in src_data, word in tgt_data
        if in_src and in_tgt:
            src_tgt.append(f'{word}\n')
        elif in_src:
            src_only.append(f'{word}\n')
        else:
            tgt_only.append(f'{word}\n')
src_tgt.append('<EOS>\n')
tgt_only.append('<BOS>\n')
src_only.append('<PAD>\n')

for langs in ('$1$2', '$2$1'):
    with open(f'data/vocab.{langs}', 'w') as vocab_file:
        tgt_range = len(src_tgt) + len(tgt_only)
        vocab_file.write(f'#2:{2 + tgt_range}\n')
        vocab_file.writelines(src_tgt)
        vocab_file.writelines(tgt_only)
        vocab_file.writelines(src_only)
    src_only, tgt_only = tgt_only, src_only
END
)"
    eval "$output"
}

echo -e "\n[8/10] Applying BPE Subword Tokenization..."
for path in "data/training" "data/validation" "data/testing"; do
    subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "$path/data.tok.$src_lang" > "$path/data.tok.bpe.$src_lang"
    subword-nmt apply-bpe -c "data/codes.$src_lang$tgt_lang" < "$path/data.tok.$tgt_lang" > "$path/data.tok.bpe.$tgt_lang"
done
cat "data/training/data.tok.bpe.$src_lang" "data/training/data.tok.bpe.$tgt_lang" \
    | subword-nmt get-vocab > "data/vocab.$src_lang$tgt_lang"
build_vocab "$src_lang" "$tgt_lang" "data/training"
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
