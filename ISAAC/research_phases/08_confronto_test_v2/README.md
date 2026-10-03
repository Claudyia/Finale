# 08 - Confronto a 3 su dataset_test_v2

Verifica indipendente dei 3 modelli principali del progetto. In tre passaggi: i primi
due su test contaminato / pulito, il terzo dopo aver rifatto la fase 06 dal baseline
ufficiale.

## Giro 3 (baseline ufficiale, fase 06 rifatta) - 2026-10-03

Il giro 2 usava per `fase06_finale` e per il "terzo modello" pesi addestrati partendo dal
vecchio `yolov8n-ppe_run_1` (non dal baseline ufficiale `yolov8n_run_1.2_classes_1_2.pt`).
La fase 06 e stata riaddestrata dal baseline ufficiale; i risultati del giro 2 per quei due
modelli sono archiviati (experiment MLflow `archivio_modello_vecchio/*`).

Stesso test pulito (1619 immagini, `test_v2_immagini_pulite_1619.txt`), stesso script
(`07_testv2/compare_models_local.py`), device `mps`:

```text
modello                    precision   recall     f1       map50    map50_95
baseline ufficiale           0.503      0.501     0.480     0.433     0.202
fold2 (fase 05)              0.797      0.447     0.447     0.439     0.220
fase06 nuova                 0.739      0.744     0.719     0.784     0.450
fase06 vecchia (archiviata)  0.767      0.735     0.726     0.788     0.456
terzo nuovo (vestplus)       0.757      0.719     0.717     0.783     0.453
terzo vecchio (archiviato)   0.744      0.715     0.701     0.770     0.468
```

OOD (55 immagini, filtro pose): baseline 63 allucinazioni, fold2 3 (-95.2%), fase06 nuova 9
(-85.7%), terzo nuovo 6 (-90.5%). Il fold2 e stato rifatto il 2026-10-03 sul test pulito (prima in MLflow era su
`dataset_test_isaac`, vecchio experiment archiviato).

Il baseline ha gli stessi numeri del giro 2, quindi era gia quello ufficiale; la fase 06
nuova e praticamente uguale alla vecchia: il punto di partenza conta poco. Fase 06 e terzo
modello sono alla pari sul test pulito. Su MLflow (http://localhost:5001) gli experiment
`baseline_vs_fold2`, `baseline_vs_fase06` e `baseline_vs_terzo_modello` contengono ciascuno i
run `base` e `ood` con le immagini; il procedimento e in `ISAAC/README.md`, sezione "Tracking MLflow". Per
caricare: `MLFLOW_TRACKING_URI=http://localhost:5001 python3 log_to_mlflow.py fase06`.

**Conclusione del giro 3:** fold2 scartato (mAP50-95 0.220, recall sotto il baseline); fase 06
e terzo modello alla pari (0.450 contro 0.453, differenze dentro il rumore: un seed, 1619
immagini, 55 OOD). Scelta consigliata: **terzo (vestplus)** per meno allucinazioni OOD (6
contro 9); la fase 06 se si privilegia la recall (0.744 contro 0.719).

Le sezioni sotto descrivono i giri 1 e 2 (storico).

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
