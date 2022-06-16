#!/bin/bash

echo "Performing Word Tokenization with Moses..."
sacremoses -l de -j 4 tokenize  < data/train.de > data/train.tokenized.de
sacremoses -l en -j 4 tokenize  < data/train.en > data/train.tokenized.en
echo "Done."

echo -e "\nLearning Model Vocabulary with BPE..."
cat data/train.tokenized.de data/train.tokenized.en | subword-nmt learn-bpe -s 18000 -o data/out
subword-nmt apply-bpe -c data/out < data/train.tokenized.de | subword-nmt get-vocab > data/vocab.de
subword-nmt apply-bpe -c data/out < data/train.tokenized.en | subword-nmt get-vocab > data/vocab.en
paste data/vocab.de data/vocab.en > data/vocab.de-en
echo "Done."

echo -e "\nPerforming Subword Tokenization with BPE..."
subword-nmt apply-bpe -c data/out --vocabulary data/vocab.de --vocabulary-threshold 50 < data/train.tokenized.de > data/train.bpe.de
subword-nmt apply-bpe -c data/out --vocabulary data/vocab.en --vocabulary-threshold 50 < data/train.tokenized.en > data/train.bpe.en
echo "Done."

echo -e "\nCombining Source and Target Training Data..."
paste data/train.bpe.de data/train.bpe.en > data/train.bpe.de-en
paste data/train.bpe.de data/train.bpe.en > data/train.bpe.en-de
echo "Done."

echo -e "\nPerforming Word Tokenization with Moses..."
sacremoses -l de -j 4 tokenize  < data/test.de > data/test.tokenized.de
sacremoses -l en -j 4 tokenize  < data/test.en > data/test.tokenized.en
echo "Done."

echo -e "\nPerforming Subword Tokenization with BPE..."
subword-nmt apply-bpe -c data/out --vocabulary data/vocab.de --vocabulary-threshold 50 < data/test.tokenized.de > data/test.bpe.de
subword-nmt apply-bpe -c data/out --vocabulary data/vocab.en --vocabulary-threshold 50 < data/test.tokenized.en > data/test.bpe.en
echo "Done."

echo -e "\nCombining Source and Target Testing Data..."
paste data/train.bpe.de data/test.bpe.en > data/test.bpe.de-en
paste data/train.bpe.en data/test.bpe.de > data/test.bpe.en-de
echo "Done."
