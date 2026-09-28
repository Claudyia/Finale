# 04 - K-Fold Validation

Questa fase sostituisce lo schema holdout fisso con una validazione k-fold.

## Perche

Prima i dataset usavano una divisione fissa:

```text
train: images/train
val: images/val
```

Questo e uno schema holdout. Con k-fold, invece, si creano piu split e si addestra un modello diverso per ogni fold.

## Scelta corretta per questo progetto

Il k-fold va applicato al dataset ID/PPE usato per il training.

Il dataset OOD deve restare separato come test esterno:

```text
isaac_odd_dataset
```

Motivo: se le immagini OOD entrano sia nel training/hard negative sia nella valutazione OOD, il risultato rischia leakage.

## Dataset sorgente

Il dataset migliore attuale e:

```text
scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost
```

## Fold creati

I 5 fold sono qui:

```text
research_phases/04_kfold_validation/folds/finetune_dataset_ood_hardneg_vestboost_k5
```

Dimensioni:

```text
totale immagini: 7139
fold_0: train=5710 val=1429
fold_1: train=5710 val=1429
fold_2: train=5712 val=1427
fold_3: train=5712 val=1427
fold_4: train=5712 val=1427
```

## Script

```text
research_phases/04_kfold_validation/scripts/make_yolo_kfolds.py
research_phases/04_kfold_validation/scripts/train_yolo_kfold.py
research_phases/04_kfold_validation/scripts/summarize_yolo_kfold.py
```

`make_yolo_kfolds.py` crea i fold YOLO senza copiare immagini. Per ogni fold genera:

```text
fold_N/
  train.txt
  val.txt
  data.yaml
```

`train_yolo_kfold.py` addestra un modello YOLO per ogni fold.

`summarize_yolo_kfold.py` calcola media e deviazione standard delle metriche migliori.


## Comando per creare 5 fold

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/04_kfold_validation/scripts/make_yolo_kfolds.py \
  --dataset scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost \
  --output-dir research_phases/04_kfold_validation/folds/finetune_dataset_ood_hardneg_vestboost_k5 \
  --folds 5 \
  --seed 42
```

## Comando per addestrare un solo fold

Utile per provare prima che tutto funzioni.

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/04_kfold_validation/scripts/train_yolo_kfold.py \
  --only-fold 0
```

## Comando per addestrare tutti i fold

Attenzione: questo lancia 5 training, quindi puo richiedere molte ore su CPU.

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/04_kfold_validation/scripts/train_yolo_kfold.py \
  --folds-dir research_phases/04_kfold_validation/folds/finetune_dataset_ood_hardneg_vestboost_k5 \
  --model models/yolov8n-ppe_run_1_classes_1_2.pt \
  --project finetune_runs/kfold \
  --name-prefix ppe_ood_hardneg_vestboost_kfold \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --lr0 0.001 \
  --patience 20 \
  --workers 2
```

## Comando per riassumere i risultati

Dopo aver completato i training:

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/04_kfold_validation/scripts/summarize_yolo_kfold.py \
  --runs-dir finetune_runs/kfold \
  --name-prefix ppe_ood_hardneg_vestboost_kfold \
  --output-csv research_phases/04_kfold_validation/kfold_summary.csv
```

