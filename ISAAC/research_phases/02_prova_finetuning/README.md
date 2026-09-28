# 02 - Prova con fine tuning

Questa fase contiene i tentativi di fine tuning fatti prima della soluzione finale.



## Script principali

Script di training:

```text
scripts/finetunig/finetuning.py
```

Script di analisi run:

```text
research_phases/02_prova_finetuning/scripts/analyze_finetuning_runs.py
```


## Dataset usato

Dataset originale di fine tuning:

```text
scripts/finetunig/dataset/finetune_dataset
```

## Run principali

Le run locali sono in:

```text
finetune_runs
```

Run importanti:

```text
finetune_runs/ppe_e50_img640_b8_lr01_pat100_base_map055
finetune_runs/ppe_e100_img800_b4_lr001_pat25_map052
finetune_runs/ppe_e80_img640_b8_lr001_pat20_oversamplevest_map047
```

## Risultato

Il fine tuning diretto non ha risolto bene il problema.

Esempi osservati:

```text
baseline e50 img640:
  final mAP50 circa 0.554

img800 lr001:
  early stopping
  mAP50 circa 0.523

oversample vest:
  mAP50 circa 0.472 finale
```

Inoltre la classe `vest` restava debole, con recall basso.

## Interpretazione

Il fine tuning semplice non basta. Serve modificare il dataset in modo mirato:

1. aggiungere esempi OOD negativi;
2. aggiungere esempi positivi di `vest`.

Questa conclusione porta alla fase 03.

## Comando analisi run

```bash
cd /Users/claudia/Desktop/isaac_odd

/usr/local/bin/python3 research_phases/02_prova_finetuning/scripts/analyze_finetuning_runs.py \
  --baseline ppe_e50_img640_b8_lr01_pat100_base_map055 \
  --candidate ppe_e80_img640_b8_lr001_pat20_oversamplevest_map047
```

## Comando training

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python scripts/finetunig/finetuning.py
```
