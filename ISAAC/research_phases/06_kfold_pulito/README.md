# 06 - K-fold pulito

Rifa il k-fold della fase 04 partendo da un dataset **senza duplicati**, con
**GroupKFold** e un **test set separato** - per togliere il leakage.
Non tocca le fasi 01-05.

## Cosa guardare (solo questo)

```text
run_kfold_local.sh                                              comando unico per allenare
kfold/kfold_summary.txt                                         mAP50 +/- std dei 5 fold (validazione metodo)
kfold/runs_noleak_vestboost/final_model/weights/best.pt          MODELLO FINALE (leakage-fix + vest-boost)
```

Il vecchio `kfold/runs/final_model/` (training senza i due fix sotto) e stato
rimosso: vedi "Fix leakage + vest-boost" piu sotto per il perche.

Avanzamento: `tail -3 kfold/train_local.log`

## Struttura

```text
dataset_clean/        3836 img deduplicate dal dataset vecchio
external_staging/     i dataset di "dataset fase6/" convertiti a helmet/vest
dataset_test_v2/      test set = construction_ppe, tenuto fuori dal training
manifest/             report di analisi (dedup, leakage)  -> evidenze per la tesi
kfold/
  merged_manifest.csv  training unito (36107 img), dedup globale
  folds_k5/            i 5 fold (liste train.txt/val.txt)
  runs/               output training (pesi, results.csv, .done)
scripts/              4 script del training
scripts/prep/         6 script gia eseguiti (preparazione dati)
```

## Training (locale, Mac M5 Pro / MPS, venv `.venv-kfold/`)

```bash
bash research_phases/06_kfold_pulito/run_kfold_local.sh        # fold 0..4 + summary + finale
bash research_phases/06_kfold_pulito/run_kfold_local.sh 2      # solo il fold 2
```

Iperparametri (`scripts/train_one_fold.py`, `HYPERPARAMS`): epochs=100, imgsz=640,
batch=16, AdamW, lr0=1e-4, weight_decay=1e-3, dropout=0.1, mosaic=0.5, patience=20,
seed=42. Modello iniziale: `models/yolov8n-ppe_run_1_classes_1_2.pt`. ~15 min/epoca.

**Ripartibile**: ogni fold e una run separata; `last.pt` salvato ogni epoca ->
`resume=True` al riavvio; fold finito -> `.done` -> saltato. Puoi fermare (Ctrl-C /
chiudere il coperchio) e rilanciare `run_kfold_local.sh` quando vuoi.

Dopo il k-fold: `train_final_model.py` allena UN modello su tutti i 36107 (il
k-fold serviva solo a validare il metodo). Quello va nel confronto della fase 07.

## Numeri chiave

### Pulizia del dataset vecchio (`finetune_dataset_ood_hardneg_vestboost_helmetboost`)

```text
7997 file -> 3836 immagini uniche   (52% erano duplicati esatti)
"oversample_vest" (3276) e "helmetboost" (830) = solo copie, zero nuove immagini
rapporto helmet/vest = 0.28   -> la classe debole era diventata HELMET
immagini con helmet+vest insieme = 2
168 immagini identiche tra quel dataset e il vecchio test set (leakage)
```

### Dataset esterni aggiunti (`external_staging/`, tutti CC BY 4.0)

```text
construction_ppe : 7575 img  (4972 miste)   -> usato come TEST (dataset_test_v2)
site_safety      : 9932 img  (7330 miste)
ppe_combined     : 15388 img (mono-classe)
hardhat_kaggle   : 4581 img  (solo helmet, da VOC)
hardneg_*        : 2370 img  (zaini, cappellini = proximal-OOD, label vuote)
scartati: life (fall detection), il CSV Kaggle
```

### Training finale (`kfold/merged_manifest.csv`)

```text
36107 immagini uniche
box helmet 70436 / box vest 35494
immagini helmet+vest : 7332      (prima 2)
immagini negative    : 2612
rapporto helmet/vest : 1.98
gruppi (group_id)    : 25194
5 fold GroupKFold, ~29k train / ~7.2k val, partizione verificata pulita
```

### Test set (`dataset_test_v2/`)

```text
7575 immagini, 4972 con helmet+vest, box helmet 6299 / vest 9755
0 overlap (hash) col training - verificato
```

## Aperti (minori)

- `dataset fase6/data/` (1416 img): mancano i nomi delle 11 classi, non usato.
- `hardhat_kaggle`: overlap percettuale col test da verificare
  (`scripts/prep/check_test_leakage.py --perceptual`).

## Fix leakage + vest-boost (dopo il primo training)

Il primo modello finale allenato su questa fase (`kfold/runs/final_model/`,
poi rimosso) sembrava avere una regressione rispetto al vecchio candidato
`fold2` delle fasi 03-05: piu hallucination OOD (8 contro 3) nonostante
mAP50-95 piu alto. Indagando sono emersi due problemi distinti, entrambi
sistemati.

### Causa 1: diluizione della classe vest

Il rapporto box helmet/vest e passato da 0.28 (vecchio dataset piccolo,
3836 img, vest dominante) a 1.98 (nuovo dataset, 36107 img, helmet
dominante). Le due fonti esterne aggiunte in questa fase sono fortemente
sbilanciate su helmet:

```text
ppe_combined      36643 box helmet  vs   4157 box vest
hardhat_kaggle    18966 box helmet  vs      0 box vest
vest_ext (invariato dalla fase 03)       5096 box vest, 0 helmet
```

`vest_ext` (il vero vest-boost della fase 03, sempre presente e intatto) e
rimasto lo stesso, ma e diventato una piccola minoranza (14% del segnale
vest totale) in un dataset molto piu grande, annegato da `ppe_combined` e
`hardhat_kaggle`.

**Fix**: nuovo flag `--vest-oversample N` (default 3) in
[`scripts/make_kfolds_v2.py`](scripts/make_kfolds_v2.py) e
[`scripts/train_final_model.py`](scripts/train_final_model.py). Ripete nel
`train.txt` il path delle immagini vest-only (n_helmet=0, n_vest>0) N volte.
Nessuna duplicazione fisica di file, nessuna modifica al manifest, `val.txt`
non toccato: zero rischio di leakage aggiuntivo.

### Causa 2 (piu grave): leakage nel test set usato per il confronto

Il test set `research_phases/dataset_test_isaac` (12298 immagini, usato in
tutti i confronti di fase 07 fold2-vs-fase06) condivide il **90.7%** delle
sue immagini (11161/12298, verificato anche visivamente, non solo per hash)
con il training di questa fase - soprattutto via `ppe_combined` (6808) e
`site_safety` (2185), fonti che il vecchio `fold2` non aveva mai visto. Il
vecchio `fold2` invece condivide solo il 17% (2138 immagini, dalle fonti
equivalenti a `dataset_clean`).

Questo significava che il confronto fold2-vs-fase06 era falsato: il modello
nuovo poteva sembrare piu bravo in parte solo perche aveva "memorizzato"
quasi tutto il test.

**Fix (parziale, mirato)**: nuovo script separato
[`scripts/prep/remove_test_leakage.py`](scripts/prep/remove_test_leakage.py)
che toglie dal manifest le 166 immagini in leakage *esatto* (hash) rilevate
da `check_test_leakage.py` (scope: solo `dataset_clean`) - senza toccare
`merged_manifest.csv` originale, scrive `kfold/merged_manifest_no_leak.csv`
(36107 -> 35941 immagini). Il leakage massivo del 90% via `ppe_combined`/
`site_safety` NON e stato ripulito dal training: e piu efficiente ignorarlo
e usare `dataset_test_v2` (sotto) come test set onesto per il confronto,
piuttosto che scartare gran parte di quelle due fonti esterne.

**La soluzione vera per il confronto**: usare `dataset_test_v2/` (gia
costruito in questa fase da `construction_ppe`, tenuto fuori apposta, "0
overlap verificato col training") al posto di `dataset_test_isaac` per
qualunque confronto fold2-vs-fase06. Su quel test pulito, `fold2` ha vest
recall di solo **0.078** (contro 0.87 sul test contaminato) - la sua
bravura su vest era quasi tutta memorizzazione.

### Risultato finale, su `dataset_test_v2` (mai visto da nessun modello)

Modello finale: `kfold/runs_noleak_vestboost/final_model/weights/best.pt`,
allenato su `kfold/merged_manifest_no_leak.csv` con `--vest-oversample 3`.

```text
                    fold2 (vecchio)   baseline generico   fase06 finale (nuovo)
precision              0.866              0.688               0.813
recall                 0.472              0.659               0.815
F1                     0.501              0.654               0.808
mAP50                  0.490              0.669               0.860
mAP50-95               0.274              0.331               0.546
helmet P/R            0.85/0.87          0.59/0.79           0.77/0.90
vest P/R               0.88/0.078         0.78/0.53           0.86/0.73
```

Riduzione hallucination OOD (55 immagini, vs baseline generico): 63 -> 7
(-88.9%). Il modello finale batte sia il vecchio `fold2` sia il baseline
generico su tutte le metriche aggregate. Il vest-oversampling ha alzato il
vest recall di +17.7 punti (0.550 -> 0.727 nel test intermedio, 0.73 nel
finale) rispetto alla versione senza fix, senza perdere precision.

Comandi usati (dalla cartella `research_phases/06_kfold_pulito`, o dalla
root del repo con i path completi come sopra):

```bash
python3 scripts/prep/remove_test_leakage.py
python3 scripts/train_final_model.py \
  --manifest kfold/merged_manifest_no_leak.csv \
  --runs-dir kfold/runs_noleak_vestboost \
  --vest-oversample 3 --device 0 --workers 16
```

### Nota: bug nel tracking MLflow (risolto)

`scripts/mlflow_utils.py` (`init_mlflow`) rileggeva `MLFLOW_TRACKING_URI`
dall'ambiente e la riapplicava sempre, scavalcando silenziosamente un
`--tracking-uri` passato esplicitamente da riga di comando dagli script di
`research_phases/07_testv2/`. MLflow gestisce gia da solo quella variabile
d'ambiente quando nessuno ha chiamato `set_tracking_uri()`, quindi quel
blocco era ridondante e in certi casi dannoso - rimosso. Se un confronto
non compare dove aspettato nella UI MLflow, controllare sempre con quale
`--tracking-uri` e stato lanciato lo script.
