# Neural Machine Translation in PyTorch
**Ken Sible | [NLP Group](https://nlp.nd.edu)**<br>
**University of Notre Dame**

```
usage: main.py [-h] [--seed SEED] {train,score} ...

optional arguments:
  -h, --help     show this help message and exit
  --seed SEED    random seed

subcommands:
  {train,score}
    train        train model
    score        score model
```

`usage: preprocess.sh [-ce] MERGE_OPS SRC_LANG TGT_LANG` ([WMT16](https://www.statmt.org/wmt16/))

## Train Model
```
usage: main.py train [-h] --lang LANG LANG [--data FILE FILE] [--vocab FILE] [--config FILE] [--save FILE]

optional arguments:
  -h, --help        show this help message and exit
  --lang LANG LANG  source/target language
  --data FILE FILE  training/validation data
  --vocab FILE      vocab (from BPE)
  --config FILE     model config
  --save FILE       save state_dict
```

## Score Model
```
usage: main.py score [-h] --lang LANG LANG [--data FILE] [--vocab FILE] [--config FILE] [--load FILE] [--out FILE]

optional arguments:
  -h, --help        show this help message and exit
  --lang LANG LANG  source/target language
  --data FILE       testing data
  --vocab FILE      vocab (from BPE)
  --config FILE     model config
  --load FILE       load state_dict
  --out FILE        store output
```

## Translate Input
```
usage: translate.py [-h] --lang LANG LANG [--vocab FILE] [--codes FILE] [--config FILE] [--load FILE] STRING

positional arguments:
  STRING            input string

optional arguments:
  -h, --help        show this help message and exit
  --lang LANG LANG  source/target language
  --vocab FILE      vocab (from BPE)
  --codes FILE      codes (from BPE)
  --config FILE     model config
  --load FILE       load state_dict
```
