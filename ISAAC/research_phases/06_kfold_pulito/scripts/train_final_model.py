# Step F - allena il MODELLO FINALE su tutto il training (36107 immagini).
#
# Il k-fold serviva a validare il metodo (media +/- std). Il modello che va in
# tesi e questo, addestrato su tutti i dati non-test, con gli stessi iperparametri.
#
# Split 95/5 per gruppo solo per monitorare l'early stopping (il 5% resta
# comunque dati di training, non e un test).
#
# Ripartibile come train_one_fold.py (.done / resume).
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kfolds_v2 import build_image_profiles, class_profile, oversample_vest, relocate  # noqa: E402
from train_one_fold import HYPERPARAMS, best_row  # noqa: E402


def build_split(
    manifest: Path, dataset_root: Path, out_dir: Path, val_frac: float, seed: int, vest_oversample: int = 1,
) -> Path:
    rows = list(csv.DictReader(manifest.open(newline="")))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    by_profile: dict[str, list[str]] = defaultdict(list)
    for gid, members in groups.items():
        prof = min(
            (class_profile(int(m["n_helmet"]), int(m["n_vest"]), int(m["is_empty_label"])) for m in members),
            key=lambda p: {"helmet+vest": 0, "helmet": 1, "vest": 2, "negative": 3}[p],
        )
        by_profile[prof].append(gid)

    rng = random.Random(seed)
    val_groups: set[str] = set()
    for prof, gids in sorted(by_profile.items()):
        rng.shuffle(gids)
        n_val = max(1, int(len(gids) * val_frac))
        val_groups.update(gids[:n_val])

    train_imgs, val_imgs = set(), set()
    for gid, members in groups.items():
        target = val_imgs if gid in val_groups else train_imgs
        for m in members:
            target.add(relocate(m["image_path"], dataset_root))
    val_imgs -= train_imgs

    profiles = build_image_profiles(rows, dataset_root)
    train_imgs_boosted = oversample_vest(sorted(train_imgs), profiles, vest_oversample)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(str(p) for p in train_imgs_boosted) + "\n")
    (out_dir / "val.txt").write_text("\n".join(str(p) for p in sorted(val_imgs)) + "\n")
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: train.txt\nval: val.txt\n\nnames:\n  0: helmet\n  1: vest\n"
    )
    print(
        f"Split finale: train={len(train_imgs)} (+boost={len(train_imgs_boosted)}) "
        f"val(monitor)={len(val_imgs)}"
    )
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Step F: modello finale su tutto il training.")
    parser.add_argument("--manifest", default="research_phases/06_kfold_pulito/kfold/merged_manifest.csv")
    parser.add_argument("--dataset-root", default="research_phases/06_kfold_pulito")
    parser.add_argument("--model", default="models/yolov8n-ppe_run_1_classes_1_2.pt")
    parser.add_argument("--runs-dir", default="research_phases/06_kfold_pulito/kfold/runs")
    parser.add_argument("--name", default="final_model")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument(
        "--vest-oversample",
        type=int,
        default=3,
        help="Ripete le immagini vest-only (no helmet) N volte nel train.txt finale, "
        "per contrastare la diluizione da ppe_combined/hardhat_kaggle. 1 = disattivato.",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--mlflow-uri", default="file:research_phases/06_kfold_pulito/kfold/mlruns")
    parser.add_argument("--experiment", default="final_model_06")
    args = parser.parse_args()

    if args.epochs:
        HYPERPARAMS["epochs"] = args.epochs

    runs_dir = Path(args.runs_dir).resolve()  # assoluto: vedi nota in train_one_fold.py
    run_dir = runs_dir / args.name
    done_marker = run_dir / ".done"
    last_ckpt = run_dir / "weights" / "last.pt"

    from ultralytics import YOLO

    if done_marker.exists():
        print(f"{args.name}: gia completato. Salto.")
        return

    data_yaml = build_split(
        Path(args.manifest), Path(args.dataset_root),
        run_dir.parent / f"{args.name}_split", args.val_frac, HYPERPARAMS["seed"],
        vest_oversample=args.vest_oversample,
    )

    common = dict(
        data=str(Path(data_yaml).resolve()), project=str(runs_dir), name=args.name,
        exist_ok=True, workers=args.workers, **HYPERPARAMS,
    )
    if args.device:
        common["device"] = args.device

    if last_ckpt.exists():
        print(f"{args.name}: RESUME da {last_ckpt}")
        YOLO(str(last_ckpt)).train(resume=True)
    else:
        print(f"{args.name}: training da {args.model}")
        YOLO(args.model).train(**common)

    results_csv = run_dir / "results.csv"
    row = best_row(results_csv) if results_csv.exists() else {}
    metrics = {
        "best_epoch": row.get("epoch", float("nan")),
        "precision": row.get("metrics/precision(B)", float("nan")),
        "recall": row.get("metrics/recall(B)", float("nan")),
        "map50": row.get("metrics/mAP50(B)", float("nan")),
        "map50_95": row.get("metrics/mAP50-95(B)", float("nan")),
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    done_marker.write_text("ok\n")
    print(f"{args.name} FATTO: {metrics}")
    print(f"Pesi finali: {run_dir / 'weights' / 'best.pt'}")

    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=args.name, tags={"pipeline": "training", "phase": "06_final"}):
            mlflow.log_params({**HYPERPARAMS, "model": args.model, "train_images": "all_non_test"})
            mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})
            best = run_dir / "weights" / "best.pt"
            if best.exists():
                mlflow.log_artifact(str(best), artifact_path="weights")
        print("MLflow: modello finale registrato.")
    except Exception as exc:  # noqa: BLE001
        print(f"(MLflow ignorato: {exc})")


if __name__ == "__main__":
    main()
