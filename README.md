# Neural Machine Translation System

```
usage: translate.py [-h] [--seed SEED] {train,score,input} ...

optional arguments:
  -h, --help          show this help message and exit
  --seed SEED         random seed

subcommands:
  {train,score,eval}
    train             train model
    score             score model
    input             translate input
```

## Train Model
```
usage: translate.py train [-h] --data FILE --val FILE --langs LANG LANG
                          --vocab FILE [--save FILE]

optional arguments:
  -h, --help         show this help message and exit
  --data FILE        training data
  --val FILE         validation data
  --langs LANG LANG  source/target language
  --vocab FILE       model vocabulary
  --save FILE        save state_dict
```

## Score Model
```
usage: translate.py score [-h] --data FILE --langs LANG LANG --vocab FILE
                          --load FILE --out FILE

optional arguments:
  -h, --help         show this help message and exit
  --data FILE        test data
  --langs LANG LANG  source/target language
  --vocab FILE       model vocabulary
  --load FILE        load state_dict
  --out FILE         save score/output
```

## Translate Input
```
usage: translate.py input [-h] --langs LANG LANG --vocab FILE --codes FILE
                          --load FILE
                          string

positional arguments:
  string             input string

optional arguments:
  -h, --help         show this help message and exit
  --langs LANG LANG  source/target language
  --vocab FILE       model vocabulary
  --codes FILE       BPE codes file
  --load FILE        load state_dict
```
