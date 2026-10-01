#codice per iterare sui fold creati dallo script make_yolo_kfolds.py,
# e per ogni fold, addestrare un modello YOLO usando il file data.yaml del
# fold come input. I risultati di ogni fold saranno salvati in una directory separata. 
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def iter_fold_data(folds_dir: Path) -> list[Path]:
    """_funzione che prende il percorso della directory dei fold e restituisce una
    lista di percorsi dei file data.yaml per ogni fold.
    I file data.yaml sono quelli creati dallo script make_yolo_kfolds.py,
    e contengono le informazioni sui percorsi delle immagini di train e val per ogni fold.
    """
    return sorted(folds_dir.glob("fold_*/data.yaml"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train un modello YOLO per ogni split k-fold.")
    parser.add_argument(
        "--folds-dir",
        default="research_phases/04_kfold_validation/folds/finetune_dataset_ood_hardneg_vestboost_k5",
    )
    parser.add_argument("--model", default="yolov8n_run_1.2_classes_1_2.pt")
    parser.add_argument("--project", default="finetune_runs/kfold")
    parser.add_argument("--name-prefix", default="ppe_ood_hardneg_vestboost_kfold")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="", help="Use 0 for Colab GPU, cpu for CPU, or leave empty for auto.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--exist-ok", action="store_true", help="Allow YOLO to reuse an existing run directory.")
    parser.add_argument(
        "--only-fold",
        type=int,
        default=-1,
        help="Train solo un   fold. Usea-1 per trainare tutti i fold.",
    )
    args = parser.parse_args()

    folds_dir = Path(args.folds_dir)
    fold_data_files = iter_fold_data(folds_dir)
    if not fold_data_files:
        raise FileNotFoundError(f"nessun file data.yaml trovato  in: {folds_dir}")

    for fold_data in fold_data_files:
        fold_name = fold_data.parent.name
        fold_index = int(fold_name.split("_")[-1])
        if args.only_fold >= 0 and fold_index != args.only_fold:
            continue

        run_name = f"{args.name_prefix}_{fold_name}"
        print(f"Training {fold_name}")
        print(f"Data: {fold_data}")
        print(f"Run: {Path(args.project) / run_name}")

        model = YOLO(args.model)
        train_kwargs = {
            "data": str(fold_data),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "patience": args.patience,
            "project": args.project,
            "name": run_name,
            "workers": args.workers,
            "exist_ok": args.exist_ok,
        }
        if args.device:
            train_kwargs["device"] = args.device
        model.train(
            **train_kwargs,
        )


if __name__ == "__main__":
    main()
