# Neural Machine Translation in PyTorch
**Ken Sible | [NLP Group](https://nlp.nd.edu)** | **University of Notre Dame**

Note, any option in `config.toml` can also be passed as a command line argument,
```
$ python translate.py --model model.deen.pt --beam-size 10 "Guten Tag!"
```

and any output from `stdout` can be diverted using the output redirection operator.
```
$ python translate.py --model model.deen.pt --file input.de > output.en
```

## Train Model
```
usage: main.py [-h] --lang SRC TGT --data FILE --test FILE --vocab FILE --codes FILE --model FILE --config FILE --log FILE [--seed SEED] [--tqdm]

options:
  -h, --help      show this help message and exit
  --lang SRC TGT  language pair
  --data FILE     training data
  --test FILE     testing data
  --vocab FILE    vocab file (shared)
  --codes FILE    codes file (shared)
  --model FILE    model file (.pt)
  --config FILE   config file (.toml)
  --log FILE      log file (.md)
  --seed SEED     random seed
  --tqdm          import tqdm
```

## Score Model
```
usage: score.py [-h] --data FILE --model FILE [--tqdm]

options:
  -h, --help    show this help message and exit
  --data FILE   testing data
  --model FILE  model file (.pt)
  --tqdm        import tqdm
```

## Translate Input
```
usage: translate.py [-h] --model FILE (--string STRING | --file FILE)

options:
  -h, --help       show this help message and exit
  --model FILE     model file (.pt)
  --string STRING  input string
  --file FILE      input file
```

## Model Configuration (Default)
```
embed_dim           = 512   # dimensions of embedding sublayers
ff_dim              = 2048  # dimensions of feed-forward sublayers
num_heads           = 8     # number of parallel attention heads
dropout             = 0.1   # dropout for emb/ff/attn sublayers
num_layers          = 6     # number of encoder/decoder layers
max_epochs          = 250   # maximum number of epochs, halt training
lr                  = 3e-4  # learning rate (step size of the optimizer)
patience            = 3     # number of epochs tolerated w/o improvement
decay_factor        = 0.8   # if patience reached, lr *= decay_factor
min_lr              = 5e-5  # minimum learning rate, halt training
label_smoothing     = 0.1   # label smoothing (regularization technique)
clip_grad           = 1.0   # maximum allowed value of gradients
batch_size          = 4096  # number of tokens per batch (source/target)
max_length          = 256   # maximum sentence length (during training)
beam_size           = 5     # beam search decoding (length normalization)
```
