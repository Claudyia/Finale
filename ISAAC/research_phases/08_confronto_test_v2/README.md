# 08 - Confronto a 3 su dataset_test_v2

Verifica indipendente dei 3 modelli principali del progetto. In due passaggi,
perche' il primo giro si e' rivelato contaminato.

## Giro 1 (CONTAMINATO) - 2026-09-28

Confronto sulle 7575 immagini di `dataset_test_v2`, ritenuto "0 overlap
verificato col training" dal README di fase 6. Verificato pero' solo per
hash esatto (SHA1) delle immagini.

## Controllo di leakage percettuale (average-hash, non solo SHA1)

Girato `check_test_leakage.py --perceptual` (train = `merged_manifest_no_leak.csv`,
36107 immagini; test = `dataset_test_v2`, 7575 immagini). Risultato:

```
Match esatti (SHA1)               : 0
Quasi-duplicati (Hamming <= 4)    : 5956  su 7575  (79%)
  di cui distanza 0 (hash identico): 3647
```

Verificato visivamente (non solo per numero) su 3 coppie a distanza 0 e 4:
tutte la stessa foto stock, solo ritagliata/ricompressa/ruotata diversamente
(SHA1 diverso, contenuto identico). Non falsi positivi dell'hash.

`dataset_test_v2` **non era pulito**: il 79% delle sue immagini sono in
realta' copie (ricompresse/ritagliate) di foto gia' nel training, soprattutto
via `site_safety` e `ppe_combined`. Report completo:
`metriche/leakage_report.csv` (5956 righe).

File dei risultati del giro 1, tenuti per trasparenza (suffisso `_CONTAMINATO`):
`metriche/confronto_baseline_vs_fold2_CONTAMINATO.json`,
`metriche/confronto_baseline_vs_fase06_CONTAMINATO.json`,
`metriche/riepilogo_CONTAMINATO.csv`.

## Giro 2 (PULITO) - stesso giorno

Tolte le 5956 immagini in leakage, restano **1619 immagini davvero mai viste**
da nessuno dei tre modelli (lista: `metriche/test_v2_immagini_pulite_1619.txt`).
Stesso script (`compare_models_local.py`), stessi modelli, ripuntato su
questo sottoinsieme.

```
modello                       precision   recall     f1       map50    map50_95
baseline_yolov8n_run_1.2        0.503      0.501      0.480     0.433     0.202
fold2_ppe_run12                   0.797      0.447      0.447     0.439     0.220
fase06_finale                     0.767      0.735      0.726     0.788     0.456
```

(CSV: `metriche/riepilogo_PULITO.csv`, JSON grezzi:
`metriche/confronto_baseline_vs_fold2_PULITO.json`,
`metriche/confronto_baseline_vs_fase06_PULITO.json`)

## Cosa cambia rispetto al giro 1

Tutti i numeri scendono (il leakage gonfiava tutti e 3 i modelli, non solo
fase06), ma **l'ordine di classifica resta lo stesso**:

- **fold2 resta peggio del baseline** su recall e f1 (0.447 contro 0.501 e
  0.480) - non era un artefatto del leakage, e' un problema reale del
  modello.
- **fase06_finale resta nettamente il migliore**, e batte il baseline su
  tutto per un margine enorme (recall +46.6%, mAP50 +82%, mAP50-95 +126%) -
  ma con margini piu' piccoli e piu' credibili di quelli (gonfiati) del
  giro 1 (es. mAP50-95: 0.546 contaminato -> 0.456 pulito).

Conclusione operativa invariata: **fase06_finale e' il modello da usare**,
ora pero' su basi solide, non contaminate.

## Modelli

| Nome cartella                | File modello                                                                          | Fase   |
|-------------------------------|-----------------------------------------------------------------------------------------|--------|
| baseline_yolov8n_run_1.2      | `yolov8n_run_1.2_classes_1_2.pt`                                                         | 01-04  |
| fold2_ppe_run12                | `ppe_run12_yoloauto_fold2_best.pt`                                                        | 04     |
| fase06_finale                  | `research_phases/06_kfold_pulito/kfold/runs_noleak_vestboost/final_model/weights/best.pt` | 06     |

## Cartelle

```
metriche/    JSON + CSV di entrambi i giri (suffisso _CONTAMINATO / _PULITO),
             leakage_report.csv, lista delle 1619 immagini pulite
immagini/    confusion matrix, curve P/R/F1/PR, batch di validazione per
             ciascun modello, in entrambi i giri
ood_check_124img/  tentativo di allargare il test OOD (fallito: il pool piu'
             grande contiene 70/124 file corrotti, vedi nota in chat)
```

## Aperto

- `dataset_test_isaac` (usato in fase 07/07v2) non e' mai stato controllato
  per leakage percettuale, solo `dataset_test_v2` qui. Probabile che abbia
  lo stesso problema o peggio (condivide gia' il 90.7% delle immagini via
  hash esatto/fonte con fase 06, per SHA1 diretto - vedi fase 06 README).
- Il set OOD resta fermo a 55 immagini (vedi cartella `ood_check_124img/`),
  nessun modo di allargarlo con i file presenti in questa copia del progetto.
