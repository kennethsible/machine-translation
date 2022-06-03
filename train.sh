#!/bin/bash

python translate.py --train data/train.bpe.de-en -o output --save model
# python translate.py data/test.bpe.de-en -o output --load model
