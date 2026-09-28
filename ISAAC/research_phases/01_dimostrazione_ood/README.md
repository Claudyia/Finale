# 01 - Dimostrazione dell'esistenza del problema OOD

Questa fase dimostra che il modello PPE originale produce detection false su immagini fuori distribuzione.

La valutazione corretta usa anche il modello pose/persona, perche la pipeline ISaAC non dovrebbe considerare una detection PPE se non e coerente con una persona rilevata.

```

## Dataset usati

Immagini ID:

```text
scripts/finetunig/dataset/finetune_dataset/images/val
scripts/finetunig/dataset/finetune_dataset/labels/val
```

Immagini OOD:

```text
isaac_odd_dataset
```

## Script

Script principale:

```text
research_phases/01_dimostrazione_ood/scripts/eval_yolo_ood_hallucinations.py
```


## Report

Report usato:

```text
research_phases/01_dimostrazione_ood/reports/baseline_with_pose/yolo_ood_hallucinations.csv
```


```

## Risultato baseline con pose/persona

```text
ID true positive: 56
ID false positive: 121
OOD hallucinations: 384
OOD mean confidence: 0.605
.

## Comando


cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/01_dimostrazione_ood/scripts/eval_yolo_ood_hallucinations.py \
  --model models/yolov8n-ppe_run_1_classes_1_2.pt \
  --pose-model models/yolov8n-pose.pt \
  --require-pose-context \
  --id-images-dir scripts/finetunig/dataset/finetune_dataset/images/val \
  --id-labels-dir scripts/finetunig/dataset/finetune_dataset/labels/val \
  --ood-images-dir isaac_odd_dataset \
  --output-dir research_phases/01_dimostrazione_ood/reports/baseline_with_pose \
  --id-limit 100 \
  --ood-limit 300
```
