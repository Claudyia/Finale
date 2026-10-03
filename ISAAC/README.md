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
06 - K-fold pulito (dataset senza duplicati/leakage)
07 / 07v2 - Confronto baseline vs modello migliore su dataset di test
08 - Confronto a 3 modelli su test pulito + tracking MLflow (Docker)
```

## Regola sul baseline (leggere prima di tutto)

Il **baseline ufficiale** e uno solo ed e sempre lo stesso file:

```text
yolov8n_run_1.2_classes_1_2.pt          (sha1 171cf5cd8ee43f611f3546affa6c191d5109ec50)
```

- Ogni confronto (fasi 01, 07v2, 08, MLflow) usa questo baseline.
- Ogni training dalla fase 04 in poi parte da questo file (`--model` di default negli script).
- Il vecchio `models/yolov8n-ppe_run_1_classes_1_2.pt` (sha1 `86726f98...`, addestrato
  da `yolov8n.pt` il 2025-05-27) **e stato rimosso**: era un modello diverso e usarlo
  ha prodotto risultati non confrontabili. Non reintrodurlo. Resta nella cronologia git.
- I risultati ottenuti con il vecchio file (fase 07 originale, fase 06 del 2026-09-10/28,
  experiment MLflow `archivio_modello_vecchio/*`) sono **archiviati, non validi** per i confronti.

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
    08_confronto_test_v2/
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
    Store MLflow locali (esperimenti precedenti). I confronti attuali (fase 08) stanno
    nel server MLflow in Docker, `docker/mlflow` (http://localhost:5001).
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

Risultato baseline (rifatto il 2026-10-03 col baseline ufficiale; il report col vecchio
modello, 56 / 121 / 384, e in `reports/archivio_modello_vecchio_baseline_with_pose/`):

```text
ID true positive: 55
ID false positive: 142
OOD hallucinations: 405
OOD mean confidence: 0.450
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

Il candidato `ppe_run12_yoloauto_fold2_best.pt` usato nei confronti **viene da questa
fase (fase 2, fold_2 di `vestboost_k5_exp2`), non dalla 04**. Metadati letti dal file:
addestrato il 2026-07-10, 100 epoche, partito da `yolov8n_run_1.2_classes_1_2.pt`
(baseline ufficiale), quindi valido. La fase 04 produce solo i fold e non ha pesi nel repo.

## Fase 06 - K-fold pulito

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

Stato: **rifatta il 2026-10-02/03 dal baseline ufficiale** `yolov8n_run_1.2_classes_1_2.pt`
(prima era partita dal vecchio modello, vedi "Regola sul baseline"). Training sulla
macchina GPU (NVIDIA L40, venv `ISAAC/.venv`, `--device 0 --workers 16`), 100 epoche,
seed 42. Il modello finale ha impiegato 5.9 ore.

Validazione dei 5 fold (`kfold/kfold_summary.txt`, media +/- deviazione standard):

```text
precision  0.8416 +/- 0.0022
recall     0.8005 +/- 0.0114
mAP50      0.8735 +/- 0.0083
mAP50-95   0.5730 +/- 0.0082
```

Modello finale (`kfold/runs_noleak_vestboost/final_model/weights/best.pt`), validazione
interna (5% dei dati tenuto da parte, NON e il test): P 0.829, R 0.821, mAP50 0.881,
mAP50-95 0.585 (helmet mAP50-95 0.494, vest 0.677). Ricetta: manifest
`merged_manifest_no_leak.csv` (35941 img, tolte le 166 in leakage esatto col test) e
`--vest-oversample 3`. Il test vero e la fase 08.

Un secondo modello finale, `runs_vestplus/final_model` ("terzo modello"), usa il manifest
`merged_manifest_v2_no_leak.csv` (36537 img, con `vest_ext`) e `--vest-oversample 3`.
Va riaddestrato dal baseline ufficiale: **in corso/da completare** (la versione precedente
era partita dal vecchio modello ed e archiviata).

Dettagli e comando di training: `research_phases/06_kfold_pulito/README.md`.

## Fase 07 / 07v2 - Confronto baseline vs modello migliore

Cartelle:

```text
research_phases/07_test
research_phases/07_testv2
```

Scopo: confrontare il modello baseline con il modello candidato
(`ppe_run12_yoloauto_fold2_best.pt`, dalla fase 05) sullo stesso dataset di test
(`research_phases/dataset_test_isaac`).

Script: `compare_models_local.py` (baseline vs candidate, metriche su split `test`) e
`evaluate_id_ood_comparison.py` (metriche ID + allucinazioni OOD con filtro pose).

**Fase 07 originale: risultati rimossi.** I numeri riportati qui in precedenza
(baseline P 0.569 / R 0.766 / mAP50-95 0.248) erano stati ottenuti col vecchio
`yolov8n-ppe_run_1`, non col baseline ufficiale, e non sono confrontabili. Gli script di
`07_test/` ora puntano al baseline ufficiale; il confronto valido e la 07v2.

Risultato fase 07v2 (baseline ufficiale, `dataset_test_isaac`, 12298 immagini):

```text
              baseline   candidate   delta
precision       0.421       0.725    +72.2%
recall          0.482       0.669    +38.8%
f1              0.382       0.662    +73.3%
mAP50           0.325       0.675   +108.0%
mAP50-95        0.112       0.327   +192.9%
```

Attenzione: `dataset_test_isaac` condivide il 90.7% delle immagini col training della
fase 06 (vedi `06_kfold_pulito/README.md`), quindi va bene solo per il candidato fold2;
per confrontare la fase 06 si usa il test pulito della fase 08.

## Fase 08 - Confronto a 3 modelli su test pulito + MLflow

Cartella: `research_phases/08_confronto_test_v2` (lista e metriche del test pulito) e
`research_phases/07_testv2` (script e output grezzi).

Test pulito: le 1619 immagini di `dataset_test_v2` senza quasi-duplicati col training
(controllo percettuale average-hash, Hamming <= 4; il 79% del test v2 originale era in
leakage). Lista: `06_kfold_pulito/dataset_test_v2/test_pulito_1619.txt`, yaml:
`06_kfold_pulito/dataset_test_v2/data_test_v2_clean.yaml` (il percorso `path:` nel yaml va
adattato alla propria macchina). OOD: 55 immagini
(`07_testv2/datasets/isaac_odd_dataset/ood/dataset_ood_hard_valid`).

I tre candidati, tutti contro lo stesso baseline ufficiale:

```text
fold2     ppe_run12_yoloauto_fold2_best.pt                          fase 05
fase06    06_kfold_pulito/kfold/runs_noleak_vestboost/final_model   fase 06 (rifatta)
terzo     06_kfold_pulito/kfold/runs_vestplus/final_model           fase 06 vestplus (da rifare)
```

Risultati sul test pulito (1619 immagini), baseline ufficiale:

```text
                baseline   fold2    fase06 (nuova)
precision        0.503     0.797      0.739
recall           0.501     0.447      0.744
f1               0.480     0.447      0.719
mAP50            0.433     0.439      0.784
mAP50-95         0.202     0.220      0.450
```

Fase 06 nuova: mAP50-95 0.450 contro 0.456 della versione partita dal vecchio modello,
cioe praticamente uguale. Il baseline ha esattamente gli stessi valori di prima (0.503 /
0.433 / 0.202), a conferma che era gia quello ufficiale.

OOD (55 immagini, filtro pose): allucinazioni del baseline 63 (28 immagini, 50.9%);
fold2 3 (riduzione 95.2%); fase 06 nuova 9 (8 immagini, 14.5%), riduzione 85.7%.
Il fold2 allucina meno ma ha recall 0.447 (sotto il baseline): scarta piu cose, anche quelle giuste. Sul set ID di `dataset_test_isaac`
(contaminato per la fase 06, indicativo) mAP50-95 0.112 -> 0.333.

### Tracking MLflow (Docker)

MLflow gira in Docker (`docker/mlflow`, UI su http://localhost:5001). Un experiment per
candidato, ognuno con due run: `base` (metriche su test pulito, da `compare_models_local.py`)
e `ood` (allucinazioni, da `evaluate_id_ood_comparison.py`). **Ogni run contiene anche le
immagini** (confusion matrix, curve P/R/F1/PR, batch di validazione baseline e candidate).

```text
baseline_vs_fold2                                     valido (rifatto sul test pulito, 2026-10-03)
baseline_vs_fase06                                    valido (fase 06 rifatta)
baseline_vs_terzo_modello                             da rifare dopo il training vestplus
archivio_modello_vecchio/baseline_vs_fold2            archiviato (calcolato su dataset_test_isaac, non confrontabile)
archivio_modello_vecchio/baseline_vs_fase06           archiviato (partito dal vecchio modello)
archivio_modello_vecchio/baseline_vs_terzo_modello    archiviato (idem)
```

Procedura per un candidato (dalla cartella `ISAAC/`, device `mps` su Mac):

```bash
# 1. metriche su test pulito
python3 research_phases/07_testv2/compare_models_local.py \
  --data research_phases/06_kfold_pulito/dataset_test_v2/data_test_v2_clean.yaml \
  --candidate <best.pt del candidato> \
  --project research_phases/07_testv2/comparison_runs_<nome>_pulito --device mps

# 2. allucinazioni OOD, senza creare run MLflow (--no-log)
python3 research_phases/07_testv2/evaluate_id_ood_comparison.py \
  --candidate <best.pt del candidato> \
  --output-dir research_phases/07_testv2/id_ood_comparison_runs_<nome>_new \
  --device mps --no-log
```

Poi si copiano gli output in `07_testv2/id_ood_comparison_runs/_snapshot_<nome>/`
(`comparison_report.json`, `baseline_ood_hallucinations.csv`,
`candidate_ood_hallucinations.csv`, `id_validation_baseline/`, `id_validation_candidate/`)
e si carica con `08_confronto_test_v2/log_to_mlflow.py`:

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 python3 research_phases/08_confronto_test_v2/log_to_mlflow.py fase06
```

L'argomento sceglie il padre (`fold2`, `fase06`, `terzo`); senza argomenti li ricarica
tutti e tre, quindi duplica quelli gia presenti: usare sempre l'argomento.

Nota: il client MLflow 3 dei venv locali parla con un server <3; per ispezionare gli
artifact usare l'API REST (`/api/2.0/mlflow/artifacts/list`) invece di `list_artifacts`.

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

Baseline ufficiale (unico, usato in tutte le fasi, vedi "Regola sul baseline"):

```text
yolov8n_run_1.2_classes_1_2.pt
```

Candidati, tutti partiti dal baseline ufficiale:

```text
ppe_run12_yoloauto_fold2_best.pt
  fase 05 (fase 2, fold_2), addestrato 2026-07-10. Candidato "fold2".

research_phases/06_kfold_pulito/kfold/runs_noleak_vestboost/final_model/weights/best.pt
  fase 06, rifatta il 2026-10-03 (manifest no_leak, vest-oversample 3). Candidato "fase06".

research_phases/06_kfold_pulito/kfold/runs_vestplus/final_model/weights/best.pt
  fase 06 vestplus (manifest v2 no_leak, vest-oversample 3). Candidato "terzo": da riaddestrare.
```

I fold 0-4 (`kfold/runs/fold_N`) servono solo a validare il metodo, non sono modelli da usare.
I vecchi output della 06 partiti dal modello vecchio sono in `kfold/_old/` sulla macchina
di training (non committati).

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

Su Mac (MPS):

```bash
bash research_phases/06_kfold_pulito/run_kfold_local.sh        # fold 0..4 + summary + finale
bash research_phases/06_kfold_pulito/run_kfold_local.sh 2      # solo il fold 2
```

Sulla macchina GPU, dentro `~/Finale/ISAAC`, in background (sopravvive alla chiusura di SSH;
ripartibile: i fold con `.done` vengono saltati). Prima spostare i vecchi `runs*` e `mlruns`
in `kfold/_old/`, altrimenti i `.done` fanno saltare il training:

```bash
nohup bash -c 'set -e; PY=.venv/bin/python; P=research_phases/06_kfold_pulito; \
$PY $P/scripts/make_kfolds_v2.py --manifest $P/kfold/merged_manifest.csv --dataset-root $P --out-dir $P/kfold/folds_k5; \
for k in 0 1 2 3 4; do $PY $P/scripts/train_one_fold.py --fold $k --device 0 --workers 16; done; \
$PY $P/scripts/summarize_kfold.py; \
$PY $P/scripts/train_final_model.py --manifest $P/kfold/merged_manifest_no_leak.csv \
  --runs-dir $P/kfold/runs_noleak_vestboost --vest-oversample 3 --device 0 --workers 16' \
> train_06.log 2>&1 &
```

Modello `vestplus` (terzo modello):

```bash
nohup .venv/bin/python research_phases/06_kfold_pulito/scripts/train_final_model.py \
  --manifest research_phases/06_kfold_pulito/kfold/merged_manifest_v2_no_leak.csv \
  --runs-dir research_phases/06_kfold_pulito/kfold/runs_vestplus --name final_model \
  --vest-oversample 3 --device 0 --workers 16 > train_vestplus.log 2>&1 &
```

Dettagli su iperparametri e ripartenza: `research_phases/06_kfold_pulito/README.md`.

## Come leggere il progetto

1. leggere `research_phases/01_dimostrazione_ood/README.md`;
2. leggere `research_phases/02_prova_finetuning/README.md`;
3. leggere `research_phases/03_risoluzione_hardneg_vestboost/README.md`;
4. leggere `research_phases/04_kfold_validation/README.md`;
5. leggere `research_phases/05_helmetboost_training_pipeline/README.md`;
6. leggere `research_phases/06_kfold_pulito/README.md`;
7. leggere `research_phases/08_confronto_test_v2/README.md` (test pulito e risultati).

Sequenza logica:

```text
problema OOD -> fine tuning non sufficiente -> dataset mirato (hard neg + vest boost)
  -> k-fold validation -> helmet boost -> dataset pulito senza leakage (fase 06, rifatta
  dal baseline ufficiale) -> confronto su test pulito + tracking MLflow (fase 08)
```
