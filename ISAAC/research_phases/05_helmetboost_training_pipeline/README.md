# 05 - Helmetboost Training Pipeline

Questa fase e separata dalla `04_kfold_validation`.

Obiettivo:

1. Fase 1: addestramento iniziale sul dataset helmetboost per produrre un checkpoint `best.pt`.
2. Fase 2: fine-tuning k-fold usando come modello iniziale il `best.pt` prodotto dalla fase 1.

## Fase 1 - Addestramento iniziale

Script:

```text
research_phases/05_helmetboost_training_pipeline/phase_1_initial_training/train_initial_yolo_checkpoint.py
```

Comando:

```bash
python research_phases/05_helmetboost_training_pipeline/phase_1_initial_training/train_initial_yolo_checkpoint.py \
  --data scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost_helmetboost/data.yaml \
  --model models/yolov8n-ppe_run_1_classes_1_2.pt \
  --project finetune_runs/phase_1_initial \
  --name ppe_ood_hardneg_vestboost_helmetboost_initial \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --lr0 0.0001 \
  --lrf 1 \
  --weight-decay 0.001 \
  --dropout 0.1 \
  --mosaic 0.5 \
  --perspective 0.01 \
  --patience 20 \
  --workers 2 \
  --device 0
```

Output principale:

```text
finetune_runs/phase_1_initial/ppe_ood_hardneg_vestboost_helmetboost_initial/weights/best.pt
```

## Fase 2 - Fine-tuning k-fold

Prima crea i fold:

```bash
python research_phases/05_helmetboost_training_pipeline/phase_2_kfold_finetuning/make_yolo_kfolds.py \
  --dataset scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost_helmetboost \
  --output-dir research_phases/05_helmetboost_training_pipeline/phase_2_kfold_finetuning/folds/helmetboost_k5 \
  --folds 5 \
  --seed 42
```

Poi lancia il fine-tuning k-fold partendo dal checkpoint della fase 1:

```bash
python research_phases/05_helmetboost_training_pipeline/phase_2_kfold_finetuning/train_yolo_kfold.py \
  --folds-dir research_phases/05_helmetboost_training_pipeline/phase_2_kfold_finetuning/folds/helmetboost_k5 \
  --model finetune_runs/phase_1_initial/ppe_ood_hardneg_vestboost_helmetboost_initial/weights/best.pt \
  --project finetune_runs/phase_2_kfold \
  --name-prefix ppe_ood_hardneg_vestboost_helmetboost_phase2_kfold \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --lr0 0.0001 \
  --lrf 1 \
  --weight-decay 0.001 \
  --dropout 0.1 \
  --mosaic 0.5 \
  --perspective 0.01 \
  --patience 20 \
  --workers 2 \
  --device 0
```

Per testare solo un fold:

```bash
python research_phases/05_helmetboost_training_pipeline/phase_2_kfold_finetuning/train_yolo_kfold.py \
  --folds-dir research_phases/05_helmetboost_training_pipeline/phase_2_kfold_finetuning/folds/helmetboost_k5 \
  --model finetune_runs/phase_1_initial/ppe_ood_hardneg_vestboost_helmetboost_initial/weights/best.pt \
  --only-fold 0 \
  --device 0
```
