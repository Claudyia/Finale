# ISaAC OOD / PPE Experiments

Questo progetto raccoglie gli esperimenti fatti sul modello YOLO PPE (helmet/vest)
per studiare e ridurre le detection false su immagini OOD, e per migliorare la
qualita del training con un dataset piu grande e senza leakage.

La struttura segue queste fasi:

```text
01 - Dimostrazione dell'esistenza del problema OOD
02 - Prova con fine tuning
03 - Risoluzione con hard negative OOD e vest boost
04 - Validazione k-fold
05 - Helmetboost training pipeline (initial + k-fold finetuning)
06 - K-fold pulito (dataset senza duplicati/leakage, in corso)
07 / 07v2 - Confronto baseline vs modello migliore su dataset di test
```

## Struttura principale

```text
isaac_odd originale/
  README.md

  research_phases/
    01_dimostrazione_ood/
    02_prova_finetuning/
    03_risoluzione_hardneg_vestboost/
    04_kfold_validation/
    05_helmetboost_training_pipeline/
    06_kfold_pulito/
    07_test/
    07_testv2/
    dataset_test_isaac/

  models/
    Pesi dei modelli YOLO e modelli esportati (anche versioni .torchscript / ncnn).

  scripts/
    Script generali del progetto (mlflow_utils.py, model-convert.py, ...).

  scripts/finetunig/
    Script di training YOLO.

  scripts/finetunig/dataset/
    Dataset YOLO usati per training e validazione (versioni progressive:
    finetune_dataset -> ..._ood_hardneg -> ..._vestboost -> ..._helmetboost).

  dataset fase6/
    Dataset esterni grezzi (Kaggle/Roboflow) usati come sorgente per la fase 06.

  ppe_run12_yoloauto_fold2_best.pt
  yolov8n_run_1.2_classes_1_2.pt
    Modelli candidato/baseline usati nei confronti di fase 07 (vedi sotto).

  mlflow.db, mlflow_esperimenti_claudia.db, mlruns/, mlflow-artifacts/
    Tracking MLflow degli esperimenti (vedi memoria "MLflow esperimenti_claudia").
```

## Fase 01 - Dimostrazione OOD

Cartella:

```text
research_phases/01_dimostrazione_ood
```

Scopo: dimostrare che il modello originale produce detection false su immagini OOD anche usando il filtro pose/persona.

Report principale:

```text
research_phases/01_dimostrazione_ood/reports/baseline_with_pose/yolo_ood_hallucinations.csv
```

Risultato baseline:

```text
ID true positive: 56
ID false positive: 121
OOD hallucinations: 384
```

## Fase 02 - Prova fine tuning

Cartella:

```text
research_phases/02_prova_finetuning
```

Scopo: verificare se un fine tuning diretto risolve il problema OOD.

Risultato: il fine tuning diretto non ha risolto bene il problema OOD e la classe `vest` restava debole.

Script principali:

```text
scripts/finetunig/finetuning.py
research_phases/02_prova_finetuning/scripts/analyze_finetuning_runs.py
```

## Fase 03 - Hard negative OOD + vest boost

Cartella:

```text
research_phases/03_risoluzione_hardneg_vestboost
```

Scopo: ridurre le allucinazioni OOD migliorando il dataset di training.

Metodo:

```text
1. aggiunta di immagini OOD come hard negative con label vuote
2. aggiunta di esempi positivi della classe vest
```

Risultato con filtro pose/persona:

```text
OOD hallucinations: 384 -> 123
ID false positive: 121 -> 23
ID true positive: 56 -> 86
```

Classe `vest` dopo il boost:

```text
precision: 0.869
recall:    0.825
mAP50:     0.877
mAP50-95:  0.639
```

## Fase 04 - K-fold validation

Cartella:

```text
research_phases/04_kfold_validation
```

Scopo: sostituire lo split holdout fisso con una validazione k-fold.

Sono stati creati 5 fold dal dataset finale:

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

Nota: il dataset OOD resta separato come test esterno e non viene mischiato nei fold.

## Fase 05 - Helmetboost training pipeline

Cartella:

```text
research_phases/05_helmetboost_training_pipeline
```

Scopo: aggiungere una classe `helmet` piu forte partendo da un training iniziale
e poi rifinire con k-fold.

```text
Fase 1: addestramento iniziale sul dataset helmetboost -> checkpoint best.pt
Fase 2: fine-tuning k-fold usando come modello iniziale il best.pt della fase 1
```

Dettagli e comandi: `research_phases/05_helmetboost_training_pipeline/README.md`.

## Fase 06 - K-fold pulito (in corso)

Cartella:

```text
research_phases/06_kfold_pulito
```

Scopo: rifare il k-fold della fase 04 partendo da un dataset **senza duplicati**,
con **GroupKFold** e un **test set separato**, per eliminare il leakage rilevato
nel dataset precedente. Non tocca le fasi 01-05.

Dataset unito (`kfold/merged_manifest.csv`):

```text
36107 immagini uniche
box helmet 70436 / box vest 35494
immagini helmet+vest : 7332
immagini negative    : 2612
rapporto helmet/vest : 1.98
gruppi (group_id)    : 25194
5 fold GroupKFold (~29k train / ~7.2k val ciascuno)
```

Test set separato (`dataset_test_v2/`, dataset `construction_ppe`):

```text
7575 immagini, 4972 con helmet+vest, box helmet 6299 / vest 9755
0 overlap (hash) col training - verificato
```

Stato: training in corso in locale (Mac, venv `.venv-kfold/`), fold_0 avviato;
non ci sono ancora risultati finali (`kfold_summary.txt` / `final_model`).
Dettagli e comando di training: `research_phases/06_kfold_pulito/README.md`.

## Fase 07 / 07v2 - Confronto baseline vs modello migliore

Cartelle:

```text
research_phases/07_test
research_phases/07_testv2
```

Scopo: confrontare il modello baseline con il modello candidato attuale
(`ppe_run12_yoloauto_fold2_best.pt`, dalla fase 04) sullo stesso dataset di test
(`research_phases/dataset_test_isaac`), senza passare da MLflow.

Script: `compare_models_local.py` (baseline vs candidate, metriche su split `test`).

Risultato fase 07 (baseline `yolov8n_run_1.2_classes_1_2.pt`):

```text
              baseline   candidate   delta
precision       0.569       0.725    +27.3%
recall          0.766       0.669    -12.6%
f1              0.553       0.661    +19.5%
mAP50           0.568       0.675    +18.9%
mAP50-95        0.248       0.327    +32.2%
```

Risultato fase 07v2 (baseline `yolov8n_run_1.2_classes_1_2.pt`):

```text
              baseline   candidate   delta
precision       0.421       0.725    +72.2%
recall          0.482       0.669    +38.8%
f1              0.382       0.662    +73.3%
mAP50           0.325       0.675   +108.0%
mAP50-95        0.112       0.327   +192.9%
```

In entrambi i confronti il candidato migliora mAP50-95; in 07v2 migliora anche
precision/recall/f1 rispetto a un baseline piu debole.

## Dataset

Dataset originale:

```text
scripts/finetunig/dataset/finetune_dataset
```

Dataset con hard negative OOD:

```text
scripts/finetunig/dataset/finetune_dataset_ood_hardneg
```

Dataset con vest boost:

```text
scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost
```

Dataset con helmet boost (fase 05):

```text
scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost_helmetboost
```

Dataset di test per i confronti di fase 07:

```text
research_phases/dataset_test_isaac
```

Dataset esterni grezzi per la fase 06 (Kaggle/Roboflow, CC BY 4.0):

```text
dataset fase6/
```

## Modelli

Modello baseline (fase 01-04):

```text
yolov8n_run_1.2_classes_1_2.pt
```

Modello baseline usato in fase 07v2:

```text
yolov8n_run_1.2_classes_1_2.pt
```

Modello candidato attuale (migliore, fase 04 k-fold, usato come candidate in fase 07/07v2):

```text
ppe_run12_yoloauto_fold2_best.pt
```

Nota: il training della fase 06 e ancora in corso; se completato produrra un
nuovo modello finale in `research_phases/06_kfold_pulito/kfold/runs/final_model/weights/best.pt`
da confrontare con il candidato attuale.

## Comando: creare i fold (fase 04)

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/04_kfold_validation/scripts/make_yolo_kfolds.py \
  --dataset scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost \
  --output-dir research_phases/04_kfold_validation/folds/finetune_dataset_ood_hardneg_vestboost_k5 \
  --folds 5 \
  --seed 42
```

## Comando: addestrare k-fold (fase 04)

Per provare un solo fold:

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/04_kfold_validation/scripts/train_yolo_kfold.py \
  --only-fold 0
```

Per addestrare tutti e 5 i fold:

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/04_kfold_validation/scripts/train_yolo_kfold.py
```

## Comando: valutare OOD con pose (fase 01/03)

```bash
cd /Users/claudia/Desktop/isaac_odd

/Users/claudia/.pyenv/versions/3.14.4/bin/python research_phases/01_dimostrazione_ood/scripts/eval_yolo_ood_hallucinations.py \
  --model ppe_run12_yoloauto_fold2_best.pt \
  --pose-model models/yolov8n-pose.pt \
  --require-pose-context \
  --id-images-dir scripts/finetunig/dataset/finetune_dataset/images/val \
  --id-labels-dir scripts/finetunig/dataset/finetune_dataset/labels/val \
  --ood-images-dir research_phases/07_testv2/datasets/isaac_odd_dataset \
  --output-dir research_phases/03_risoluzione_hardneg_vestboost/reports/new_eval_with_pose \
  --id-limit 100 \
  --ood-limit 300
```

## Comando: confrontare baseline vs candidato (fase 07)

```bash
cd "isaac_odd originale"

python research_phases/07_testv2/compare_models_local.py \
  --baseline yolov8n_run_1.2_classes_1_2.pt \
  --candidate ppe_run12_yoloauto_fold2_best.pt \
  --data research_phases/dataset_test_isaac/data.yaml \
  --split test
```

## Comando: k-fold pulito (fase 06)

```bash
bash research_phases/06_kfold_pulito/run_kfold_local.sh        # fold 0..4 + summary + finale
bash research_phases/06_kfold_pulito/run_kfold_local.sh 2      # solo il fold 2
```

Dettagli su iperparametri e ripartenza: `research_phases/06_kfold_pulito/README.md`.

## Come leggere il progetto

1. leggere `research_phases/01_dimostrazione_ood/README.md`;
2. leggere `research_phases/02_prova_finetuning/README.md`;
3. leggere `research_phases/03_risoluzione_hardneg_vestboost/README.md`;
4. leggere `research_phases/04_kfold_validation/README.md`;
5. leggere `research_phases/05_helmetboost_training_pipeline/README.md`;
6. leggere `research_phases/06_kfold_pulito/README.md`.

Sequenza logica:

```text
problema OOD -> fine tuning non sufficiente -> dataset mirato (hard neg + vest boost)
  -> k-fold validation -> helmet boost -> dataset pulito senza leakage (in corso)
  -> confronto baseline vs modello migliore (fase 07/07v2)
```
