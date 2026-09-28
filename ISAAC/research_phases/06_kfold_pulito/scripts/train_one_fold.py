# Step D - allena UN fold del k-fold pulito.
#
# Pensato per Colab free: tutto (pesi, checkpoint, log) va su Google Drive, e lo
# script e RIPARTIBILE.
#   - fold gia finito (marker .done)        -> salta
#   - checkpoint last.pt presente           -> riprende (resume=True)
#   - altrimenti                            -> parte dal modello iniziale
# Ultralytics salva last.pt a ogni epoca: una disconnessione costa <= 1 epoca.
#
# MLflow viene scritto SOLO a fine fold (best effort), leggendo results.csv:
# se il training si interrompe non lascia run a meta.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HYPERPARAMS = dict(
    epochs=100, imgsz=640, batch=16, optimizer="AdamW",
    lr0=1e-4, lrf=1.0, weight_decay=1e-3, dropout=0.1,
    mosaic=0.5, perspective=0.01, patience=20, seed=42, save_period=10,
)


def best_row(results_csv: Path) -> dict[str, float]:
    rows = []
    with results_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({k.strip(): (float(v) if v.strip() not in ("", "nan") else float("nan"))
                        for k, v in row.items() if k.strip()})
    key = "metrics/mAP50(B)"
    return max(rows, key=lambda r: r.get(key, float("-inf"))) if rows else {}


def log_to_mlflow(run_dir: Path, fold_name: str, args, metrics: dict) -> None:
    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001
        print(f"  (mlflow non disponibile: {exc})")
        return
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=fold_name, tags={"fold": fold_name, "pipeline": "training",
                                                        "phase": "06_kfold_pulito"}):
            mlflow.log_params({**HYPERPARAMS, "model": args.model, "data": str(args.fold_dir)})
            mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})  # scarta NaN
            for name in ("results.csv", "results.png", "confusion_matrix_normalized.png",
                         "PR_curve.png", "BoxPR_curve.png"):
                fpath = run_dir / name
                if fpath.exists():
                    mlflow.log_artifact(str(fpath))
            best = run_dir / "weights" / "best.pt"
            if best.exists():
                mlflow.log_artifact(str(best), artifact_path="weights")
        print("  MLflow: run registrata.")
    except Exception as exc:  # noqa: BLE001
        print(f"  (log MLflow fallito, ignoro: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step D: allena un fold (ripartibile).")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds-dir", default="research_phases/06_kfold_pulito/kfold/folds_k5")
    parser.add_argument("--model", default="models/yolov8n-ppe_run_1_classes_1_2.pt")
    parser.add_argument("--runs-dir", default="research_phases/06_kfold_pulito/kfold/runs",
                        help="Su Colab: una cartella dentro Google Drive.")
    parser.add_argument("--device", default="", help="0 per GPU, cpu, mps. Vuoto = auto.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None, help="Override di HYPERPARAMS['epochs'].")
    parser.add_argument("--mlflow-uri",
                        default="file:research_phases/06_kfold_pulito/kfold/mlruns")
    parser.add_argument("--experiment", default="kfold_06_pulito")
    args = parser.parse_args()

    if args.epochs:
        HYPERPARAMS["epochs"] = args.epochs

    fold_name = f"fold_{args.fold}"
    args.fold_dir = Path(args.folds_dir) / fold_name / "data.yaml"
    if not args.fold_dir.exists():
        raise FileNotFoundError(f"Manca {args.fold_dir}. Esegui prima make_kfolds_v2.py")

    # Path ASSOLUTO: con un path relativo Ultralytics lo mette sotto il suo
    # settings['runs_dir'], non dove vogliamo noi.
    runs_dir = Path(args.runs_dir).resolve()
    run_dir = runs_dir / fold_name
    done_marker = run_dir / ".done"
    last_ckpt = run_dir / "weights" / "last.pt"

    from ultralytics import YOLO

    if done_marker.exists():
        print(f"{fold_name}: gia completato ({done_marker}). Salto.")
        return

    common = dict(
        data=str(args.fold_dir.resolve()), project=str(runs_dir), name=fold_name,
        exist_ok=True, workers=args.workers, **HYPERPARAMS,
    )
    if args.device:
        common["device"] = args.device

    if last_ckpt.exists():
        print(f"{fold_name}: trovato {last_ckpt} -> RESUME")
        # Se la run e stata iniziata su un'altra macchina (Colab), il path del
        # dataset in args.yaml e sbagliato: lo correggo col data.yaml locale.
        args_yaml = run_dir / "args.yaml"
        if args_yaml.exists():
            text = args_yaml.read_text().splitlines()
            fixed = [f"data: {args.fold_dir.resolve()}" if ln.startswith("data:") else ln for ln in text]
            args_yaml.write_text("\n".join(fixed) + "\n")
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        print(f"{fold_name}: training da {args.model}")
        model = YOLO(args.model)
        model.train(**common)

    # --- fine fold ---
    results_csv = run_dir / "results.csv"
    row = best_row(results_csv) if results_csv.exists() else {}
    metrics = {
        "best_epoch": row.get("epoch", float("nan")),
        "precision": row.get("metrics/precision(B)", float("nan")),
        "recall": row.get("metrics/recall(B)", float("nan")),
        "map50": row.get("metrics/mAP50(B)", float("nan")),
        "map50_95": row.get("metrics/mAP50-95(B)", float("nan")),
    }
    (run_dir / "fold_metrics.json").write_text(json.dumps(metrics, indent=2))
    done_marker.write_text("ok\n")
    print(f"{fold_name} FATTO: {metrics}")

    log_to_mlflow(run_dir, fold_name, args, metrics)


if __name__ == "__main__":
    main()
