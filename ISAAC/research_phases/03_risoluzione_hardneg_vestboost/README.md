# 03 - Risoluzione con hard negative OOD e vest boost

Questa fase contiene la soluzione che ha funzionato meglio.

## Domanda

Possiamo ridurre le allucinazioni OOD senza usare un filtro esterno, ma migliorando il training set?

## Metodo

Sono state fatte due modifiche al dataset:

1. aggiunta di immagini OOD come hard negative, cioe immagini con label vuote;
2. aggiunta di immagini positive per la classe `vest`, perche era la classe piu debole.

Questa fase non usa un tool esterno pronto: abbiamo implementato noi il flusso con script locali.

L'idea viene dal paper:

```text
Mitigating Hallucinations in YOLO-based Object Detection Models:
A Revisit to Out-of-Distribution Detection
arXiv:2503.07330
```
invece di usare solo un filtro OOD esterno dopo il detector, si costruisce un dataset OOD semanticamente vicino agli oggetti target e lo si usa nel fine tuning per ridurre l'objectness sulle immagini fuori distribuzione.

Nel nostro caso:

```text
paper: proximal OOD + fine tuning per sopprimere hallucination
nostro progetto: immagini OOD hard negative + fine tuning YOLO PPE
```

## Script

```text
research_phases/03_risoluzione_hardneg_vestboost/scripts/build_ood_hard_negative_dataset.py
research_phases/03_risoluzione_hardneg_vestboost/scripts/convert_jhboyo_ppe_dataset.py
research_phases/03_risoluzione_hardneg_vestboost/scripts/build_vestboost_dataset.py
```

Lo script di valutazione OOD e nella fase 01, perche e lo stesso usato per dimostrare il problema:

```text
research_phases/01_dimostrazione_ood/scripts/eval_yolo_ood_hallucinations.py
```

## Dataset creati

Dataset con hard negative OOD:

```text
scripts/finetunig/dataset/finetune_dataset_ood_hardneg
```

Dataset finale con hard negative + vest boost:

```text
scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost
```

Dataset esterno usato per aggiungere esempi di vest:

```text
external_datasets/ppe_jhboyo_raw
external_datasets/ppe_jhboyo_converted
```

## Report

Hard negative:

```text
research_phases/03_risoluzione_hardneg_vestboost/reports/hardneg_with_pose/yolo_ood_hallucinations.csv
```

Hard negative + vest boost:

```text
research_phases/03_risoluzione_hardneg_vestboost/reports/hardneg_vestboost_with_pose/yolo_ood_hallucinations.csv
```

I report senza filtro pose sono stati spostati in archivio, perche non rappresentano la pipeline ISaAC:

```text
research_phases/03_risoluzione_hardneg_vestboost/reports/_archive_no_pose/
```

## Risultati

Risultati con filtro pose/persona:

```text
Baseline originale:
  ID true positive: 56
  ID false positive: 121
  OOD hallucinations: 384

Hard negative:
  ID true positive: 85
  ID false positive: 43
  OOD hallucinations: 155

Hard negative + vest boost:
  ID true positive: 86
  ID false positive: 23
  OOD hallucinations: 123
```

## Miglioramento

Rispetto alla baseline, usando la metrica con pose/persona:

```text
OOD hallucinations: 384 -> 123
ID false positive: 121 -> 23
ID true positive: 56 -> 86
```

La classe `vest` e migliorata molto:

```text
vest precision: 0.869
vest recall:    0.825
vest mAP50:     0.877
vest mAP50-95:  0.639
```

## Modello migliore

```text
/Users/claudia/Desktop/isac-Tesi/isaac-cvmod/runs/detect/finetune_runs/ppe_ood_hardneg_vestboost_e50_img640_b8-2/weights/best.pt
```

## Comando: convertire dataset vest esterno

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/03_risoluzione_hardneg_vestboost/scripts/convert_jhboyo_ppe_dataset.py \
  --raw-dir external_datasets/ppe_jhboyo_raw \
  --output-dir external_datasets/ppe_jhboyo_converted \
  --require-vest \
  --train-limit 2000 \
  --val-limit 400 \
  --test-limit 400
```

## Comando: creare dataset finale

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/03_risoluzione_hardneg_vestboost/scripts/build_vestboost_dataset.py \
  --base-dataset scripts/finetunig/dataset/finetune_dataset_ood_hardneg \
  --vest-dataset external_datasets/ppe_jhboyo_converted \
  --output-dataset scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost \
  --train-limit 2000 \
  --val-limit 400
```

## Comando: valutare il modello finale su OOD con pose

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/01_dimostrazione_ood/scripts/eval_yolo_ood_hallucinations.py \
  --model /Users/claudia/Desktop/isac-Tesi/isaac-cvmod/runs/detect/finetune_runs/ppe_ood_hardneg_vestboost_e50_img640_b8-2/weights/best.pt \
  --pose-model models/yolov8n-pose.pt \
  --require-pose-context \
  --id-images-dir scripts/finetunig/dataset/finetune_dataset/images/val \
  --id-labels-dir scripts/finetunig/dataset/finetune_dataset/labels/val \
  --ood-images-dir isaac_odd_dataset \
  --output-dir research_phases/03_risoluzione_hardneg_vestboost/reports/new_eval_with_pose \
  --id-limit 100 \
  --ood-limit 300
```

## Nota su GL-MCM

GL-MCM e stato provato, ma in questo caso non separava bene ID e OOD:

```text
ID mean score:  0.3492
OOD mean score: 0.3484
```

Per questo non e stato mantenuto come soluzione finale.
