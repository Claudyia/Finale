#!/bin/bash
# K-fold pulito in locale (Mac M5 Pro, GPU Metal / MPS).
#
# Ripartibile: i fold con marker .done vengono saltati, quelli a meta riprendono
# da last.pt. Puoi interrompere con Ctrl-C e rilanciare quando vuoi.
#
#   bash research_phases/06_kfold_pulito/run_kfold_local.sh          # tutti i fold + summary + finale
#   bash research_phases/06_kfold_pulito/run_kfold_local.sh 2        # solo il fold 2
#
set -e
cd "$(dirname "$0")/../.."          # -> isaac_odd originale

PY=.venv-kfold/bin/python
S=research_phases/06_kfold_pulito/scripts
DEVICE=mps
WORKERS=6

# fold coi path locali (idempotente)
$PY $S/make_kfolds_v2.py \
  --manifest research_phases/06_kfold_pulito/kfold/merged_manifest.csv \
  --dataset-root research_phases/06_kfold_pulito \
  --out-dir research_phases/06_kfold_pulito/kfold/folds_k5

if [ -n "$1" ]; then
  FOLDS="$1"
else
  FOLDS="0 1 2 3 4"
fi

for k in $FOLDS; do
  echo ""
  echo "===== FOLD $k  $(date +%H:%M) ====="
  $PY $S/train_one_fold.py --fold "$k" --device $DEVICE --workers $WORKERS
done

# summary (funziona anche coi fold parziali)
$PY $S/summarize_kfold.py

# modello finale (solo se hai passato tutti i fold)
if [ -z "$1" ]; then
  echo ""
  echo "===== MODELLO FINALE  $(date +%H:%M) ====="
  $PY $S/train_final_model.py \
    --manifest research_phases/06_kfold_pulito/kfold/merged_manifest.csv \
    --dataset-root research_phases/06_kfold_pulito \
    --device $DEVICE --workers $WORKERS
fi
