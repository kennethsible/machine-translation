# Neural Machine Translation in PyTorch
**Ken Sible | [NLP Group](https://nlp.nd.edu)** | **University of Notre Dame**

Note, any option in `model.config` can also be passed as a command line argument,
```
$ python translate.py --lang de en --beam_size 5 --string "..."
```

and any output from `stdout` can be diverted using the output redirection operator.
```
$ python translate.py --lang de en --file infile.txt > outfile.txt
```

## Train Model
```
usage: main.py [-h] --lang LANG LANG [--data FILE] [--test FILE] [--vocab FILE] [--config FILE] [--load FILE] [--save FILE] [--seed SEED] [--tqdm]

optional arguments:
  -h, --help        show this help message and exit
  --lang LANG LANG  source/target language
  --data FILE       training data
  --test FILE       validation data
  --vocab FILE      shared vocab
  --config FILE     model config
  --load FILE       load state_dict
  --save FILE       save state_dict
  --seed SEED       random seed
  --tqdm            toggle tqdm
```

## Score Model
```
usage: score.py [-h] --lang LANG LANG [--data FILE] [--vocab FILE] [--config FILE] [--load FILE]

optional arguments:
  -h, --help        show this help message and exit
  --lang LANG LANG  source/target language
  --data FILE       testing data
  --vocab FILE      shared vocab
  --config FILE     model config
  --load FILE       load state_dict
```

## Translate Input
```
usage: translate.py [-h] --lang LANG LANG [--vocab FILE] [--codes FILE] [--config FILE] [--load FILE] (--file FILE | --string STRING | --interactive)

optional arguments:
  -h, --help        show this help message and exit
  --lang LANG LANG  source/target language
  --vocab FILE      shared vocab
  --codes FILE      shared codes
  --config FILE     model config
  --load FILE       load state_dict
  --file FILE       input file
  --string STRING   input string
  --interactive     interactive session
```

## Model Configuration (Default)
```
embed_dim           = 512   # dimensions of embedding sublayers
ff_dim              = 2048  # dimensions of feed-forward sublayers
num_heads           = 8     # number of parallel attention heads
num_layers          = 6     # number of encoder/decoder layers
dropout             = 0.1   # dropout for feed-forward/attention sublayers
max_epochs          = 250   # maximum number of epochs (halt training)
lr                  = 3e-4  # learning rate (step size of the optimizer)
patience            = 3     # number of epochs tolerated w/o improvement
label_smoothing     = 0.1   # label smoothing (regularization technique)
batch_size          = 4096  # number of tokens per batch (source/target)
max_length          = 256   # maximum sentence length (during training)
beam_width          = 5     # beam search and length normalization
```
