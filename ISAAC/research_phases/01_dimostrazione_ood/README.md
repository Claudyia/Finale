# 01 - Dimostrazione dell'esistenza del problema OOD

Questa fase dimostra che il modello PPE originale produce detection false su immagini fuori distribuzione.

La valutazione corretta usa anche il modello pose/persona, perche la pipeline ISaAC non dovrebbe considerare una detection PPE se non e coerente con una persona rilevata.

## Dataset usati

Immagini ID:

```text
scripts/finetunig/dataset/finetune_dataset/images/val
scripts/finetunig/dataset/finetune_dataset/labels/val
```

Immagini OOD:

```text
research_phases/07_testv2/datasets/isaac_odd_dataset
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


## Risultato baseline con pose/persona

Rifatto il 2026-10-03 col **baseline ufficiale** `yolov8n_run_1.2_classes_1_2.pt`
(sha1 `171cf5cd...`). Il report precedente era stato generato col vecchio
`yolov8n-ppe_run_1`, non confrontabile, ed e archiviato in
`reports/archivio_modello_vecchio_baseline_with_pose/`.

```text
                         baseline ufficiale     vecchio modello (archiviato)
ID true positive                55                         56
ID false positive              142                        121
OOD hallucinations             405                        384
OOD mean confidence          0.450                      0.605
```

Il problema OOD e confermato col baseline ufficiale: 405 detection false su immagini
senza DPI, tutte nel gruppo `dataset_ok`. Con il baseline ufficiale ci sono piu falsi
positivi ID (142 contro 121) e piu allucinazioni, ma con confidenza media piu bassa.

## Comando

Dalla cartella `ISAAC/` (l'immagine OOD sta in `07_testv2/datasets`; se la cartella
indicata non esiste lo script non da errore e salta la parte OOD):

```bash
python3 research_phases/01_dimostrazione_ood/scripts/eval_yolo_ood_hallucinations.py \
  --model yolov8n_run_1.2_classes_1_2.pt \
  --pose-model models/yolov8n-pose.pt \
  --require-pose-context \
  --id-images-dir scripts/finetunig/dataset/finetune_dataset/images/val \
  --id-labels-dir scripts/finetunig/dataset/finetune_dataset/labels/val \
  --ood-images-dir research_phases/07_testv2/datasets/isaac_odd_dataset \
  --output-dir research_phases/01_dimostrazione_ood/reports/baseline_with_pose \
  --id-limit 100 \
  --ood-limit 300
```
