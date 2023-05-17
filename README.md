# Neural Machine Translation in PyTorch
**Ken Sible | [NLP Group](https://nlp.nd.edu)** | **University of Notre Dame**

Note, any option in `model.config` can also be passed as a command line argument,
```
$ python translate.py --load model.deen --beam-size 10 --string "..."
```

and any output from `stdout` can be diverted using the output redirection operator.
```
$ python translate.py --load model.deen --data infile.de > outfile.en
```

## Train Model
```
usage: main.py [-h] --lang SRC TGT --data FILE --test FILE --vocab FILE --codes FILE --save FILE [--seed SEED] [--tqdm]

optional arguments:
  -h, --help      show this help message and exit
  --lang SRC TGT  language pair
  --data FILE     training data
  --test FILE     validation data
  --vocab FILE    shared vocab
  --codes FILE    shared codes
  --save FILE     save model
  --seed SEED     random seed
  --tqdm          enable tqdm
```

## Score Model
```
usage: score.py [-h] --data FILE --load FILE [--tqdm]

optional arguments:
  -h, --help   show this help message and exit
  --data FILE  testing data
  --load FILE  load model
  --tqdm       enable tqdm
```

## Translate Input
```
usage: translate.py [-h] --load FILE (--file FILE | --string STRING)

optional arguments:
  -h, --help       show this help message and exit
  --load FILE      load model
  --file FILE      input file
  --string STRING  input string
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
beam_size           = 5     # beam search decoding (length normalization)
```
