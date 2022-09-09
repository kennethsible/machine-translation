# Neural Machine Translation System
**Ken Sible | University of Notre Dame | [NLP Group](https://nlp.nd.edu)**

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

`usage: preprocess.sh <src_lang> <tgt_lang>`

## Train Model
```
usage: main.py train [-h] --data FILE --val FILE --langs LANG LANG --vocab FILE [--save FILE]

optional arguments:
  -h, --help         show this help message and exit
  --data FILE        training data
  --val FILE         validation data
  --langs LANG LANG  source/target language
  --vocab FILE       vocab file
  --save FILE        save state_dict
```

## Score Model
```
usage: main.py score [-h] --data FILE --langs LANG LANG --vocab FILE --load FILE --save FILE

optional arguments:
  -h, --help         show this help message and exit
  --data FILE        test data
  --langs LANG LANG  source/target language
  --vocab FILE       vocab file
  --load FILE        load state_dict
  --save FILE         save output
```

## Translate Input
```
usage: translate.py [-h] --langs LANG LANG --vocab FILE --codes FILE --load FILE INPUT

positional arguments:
  INPUT              string (source language)

optional arguments:
  -h, --help         show this help message and exit
  --langs LANG LANG  source/target language
  --vocab FILE       vocab file
  --codes FILE       codes file
  --load FILE        load state_dict
```
