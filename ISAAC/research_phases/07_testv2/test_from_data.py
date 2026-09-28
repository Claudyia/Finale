from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from ultralytics import YOLO

# Permette di eseguire questo file direttamente dalla riga di comando e di
# importare i moduli condivisi presenti nella root del progetto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_utils import (  # noqa: E402
    log_artifact_if_active,
    log_metrics_if_active,
    optional_mlflow_run,
)


# Gerarchia MLflow:
#   esperimenti_claudia/                 (cartella experiment, raggruppata dalla UI)
#   └─ esperimenti_claudia/test_id_isaac (questo experiment)
#      └─ parent run "full_<timestamp>"
#         ├─ child run "baseline"  -> yolov8n_run_1.2_classes_1_2.pt
#         └─ child run "nostro"    -> ppe_run12_yoloauto_fold2_best.pt
EXPERIMENT_NAME = "esperimenti_claudia/test_id_isaac"

# Store MLflow dedicato ai nuovi esperimenti: non tocca 'mlflow.db' (ood-test-v)
# né gli store dei confronti precedenti. Sovrascrivibile con --tracking-uri o
# con la variabile d'ambiente MLFLOW_TRACKING_URI.
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow_esperimenti_claudia.db'}"

# Cambia questi path quando vuoi testare altri modelli o un altro dataset.
BASELINE_MODEL = Path("yolov8n_run_1.2_classes_1_2.pt")
CANDIDATE_MODEL = Path("ppe_run12_yoloauto_fold2_best.pt")
DATASET_YAML_PATH = Path("research_phases/dataset_test_isaac/data.yaml")

# Output del test (dentro 07_testv2, per non toccare le run vecchie di 07_test).
PROJECT_PATH = Path("research_phases/07_testv2/test_runs")

COMPARISON_METRICS = ("precision", "recall", "f1", "map50", "map50_95", "fitness")


def to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def class_name(names: Any, class_index: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_index, names.get(str(class_index), class_index)))
    if isinstance(names, (list, tuple)) and class_index < len(names):
        return str(names[class_index])
    return str(class_index)


def resolve_path(path: str, description: str, *, must_be_file: bool = True) -> Path:
    """Path assoluto così com'è, altrimenti relativo alla root del progetto."""

    raw = Path(path).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(PROJECT_ROOT / raw)

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved.is_file() if must_be_file else resolved.exists()):
            return resolved

    raise FileNotFoundError(
        f"{description} non trovato: {path} (cerco anche sotto {PROJECT_ROOT})"
    )


def build_metrics_report(metrics: Any, config: dict[str, Any]) -> dict[str, Any]:
    box = metrics.box
    names = getattr(metrics, "names", {})
    precision = np.asarray(getattr(box, "p", []), dtype=float)
    recall = np.asarray(getattr(box, "r", []), dtype=float)
    f1 = np.asarray(getattr(box, "f1", []), dtype=float)
    map50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    map50_95 = np.asarray(getattr(box, "maps", []), dtype=float)
    class_count = max(
        len(precision), len(recall), len(f1), len(map50), len(map50_95)
    )

    def value_at(values: np.ndarray, index: int) -> float | None:
        return float(values[index]) if index < len(values) else None

    per_class = []
    for index in range(class_count):
        per_class.append(
            {
                "class_id": index,
                "class_name": class_name(names, index),
                "precision": value_at(precision, index),
                "recall": value_at(recall, index),
                "f1": value_at(f1, index),
                "mAP50": value_at(map50, index),
                "mAP50-95": value_at(map50_95, index),
            }
        )

    return {
        "configuration": config,
        "overall": {
            "precision": float(np.mean(precision)) if precision.size else None,
            "recall": float(np.mean(recall)) if recall.size else None,
            "f1": float(np.mean(f1)) if f1.size else None,
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "fitness": float(metrics.fitness),
        },
        "per_class": per_class,
        "speed_ms_per_image": getattr(metrics, "speed", {}),
        "ultralytics_results": getattr(metrics, "results_dict", {}),
    }


def overall_metrics(report: dict[str, Any]) -> dict[str, float]:
    overall = report["overall"]
    return {
        "precision": overall["precision"] or 0.0,
        "recall": overall["recall"] or 0.0,
        "f1": overall["f1"] or 0.0,
        "map50": overall["mAP50"] or 0.0,
        "map50_95": overall["mAP50-95"] or 0.0,
        "fitness": overall["fitness"] or 0.0,
    }


def print_comparison(baseline: dict[str, float], candidate: dict[str, float]) -> None:
    print()
    print("CONFRONTO ID (dataset_test_isaac)")
    print(f"{'Metrica':<14} {'baseline':>12} {'nostro':>12} {'delta':>12} Esito")
    print("-" * 60)
    for metric in COMPARISON_METRICS:
        base = baseline[metric]
        cand = candidate[metric]
        delta = cand - base
        verdict = "MIGLIORA" if delta > 0 else "PEGGIORA/UGUALE"
        print(f"{metric:<14} {base:>12.4f} {cand:>12.4f} {delta:>+12.4f} {verdict}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valuta baseline e modello nostro su un dataset YOLO con label "
            "(solo metriche ID) e le raccoglie in un experiment MLflow."
        )
    )
    parser.add_argument("--baseline", default=str(BASELINE_MODEL))
    parser.add_argument("--candidate", default=str(CANDIDATE_MODEL))
    parser.add_argument("--data", default=str(DATASET_YAML_PATH))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="", help="Esempi: cpu, mps, 0.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default=str(PROJECT_PATH))
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=(
            "URI dello store MLflow. Priorità: questo argomento > "
            "MLFLOW_TRACKING_URI > default dedicato "
            f"({DEFAULT_TRACKING_URI})."
        ),
    )
    parser.add_argument(
        "--parent-run-name",
        default=None,
        help="Nome della parent run. Default: full_<timestamp>.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help=(
            "Calcola le metriche, salva i report JSON in locale e stampa il "
            "confronto SENZA creare alcuna run MLflow."
        ),
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def evaluate_one(
    *,
    run_name: str,
    role: str,
    model_path: Path,
    data_path: Path,
    project_path: Path,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Esegue la validazione di un modello e, se attiva, la logga su MLflow."""

    val_kwargs: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "workers": args.workers,
        "project": str(project_path),
        "name": run_name,
        "exist_ok": args.exist_ok,
        "plots": True,
        "save_json": True,
    }
    if args.device:
        val_kwargs["device"] = args.device

    config = {
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": args.device or "automatic",
        "workers": args.workers,
    }

    print()
    print(f"Run '{run_name}' ({role}): {model_path}")
    metrics = YOLO(str(model_path)).val(**val_kwargs)

    output_dir = Path(metrics.save_dir)
    report = build_metrics_report(metrics, config)
    report_path = output_dir / "test_metrics.json"
    report_path.write_text(
        json.dumps(to_json_value(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    scores = overall_metrics(report)
    log_metrics_if_active(scores)
    log_artifact_if_active(output_dir, artifact_path="ultralytics-results")

    print(f"  mAP50={scores['map50']:.4f}  mAP50-95={scores['map50_95']:.4f}")
    print(f"  report: {report_path}")
    return scores


def main() -> None:
    args = parse_args()

    tracking_uri = (
        args.tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or DEFAULT_TRACKING_URI
    )
    if not args.no_log:
        mlflow.set_tracking_uri(tracking_uri)

    baseline_path = resolve_path(args.baseline, "Modello baseline")
    candidate_path = resolve_path(args.candidate, "Modello nostro")
    data_path = resolve_path(args.data, "Dataset YAML")
    project_path = Path(args.project).expanduser()
    if not project_path.is_absolute():
        project_path = PROJECT_ROOT / project_path
    project_path = project_path.resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    parent_run_name = (
        args.parent_run_name
        or f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    roles = (
        ("baseline", "baseline", baseline_path),
        ("nostro", "candidate", candidate_path),
    )

    print(
        f"Tracking URI: {tracking_uri}"
        f"{' (non usato: --no-log)' if args.no_log else ''}"
    )
    print(f"Experiment: {args.experiment}")
    print(f"Parent run: {parent_run_name}")
    print(f"Log su MLflow: {'NO (--no-log)' if args.no_log else 'sì'}")
    print(f"Dataset: {data_path}  (split={args.split})")
    print(f"Baseline: {baseline_path}")
    print(f"Nostro:   {candidate_path}")

    common_params = {
        "dataset": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": args.device or "automatic",
        "workers": args.workers,
    }

    results: dict[str, dict[str, float]] = {}

    with optional_mlflow_run(
        enabled=not args.no_log,
        experiment_name=args.experiment,
        run_name=parent_run_name,
        params={
            **common_params,
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
        },
        tags={
            "kind": "parent",
            "pipeline": "ultralytics-validation",
            "task": "object-detection",
        },
    ):
        for run_name, role, model_path in roles:
            with optional_mlflow_run(
                enabled=not args.no_log,
                experiment_name=args.experiment,
                run_name=run_name,
                params={**common_params, "role": role, "model": str(model_path)},
                tags={
                    "kind": "child",
                    "model_role": role,
                    "pipeline": "ultralytics-validation",
                    "task": "object-detection",
                },
                nested=True,
            ):
                results[role] = evaluate_one(
                    run_name=run_name,
                    role=role,
                    model_path=model_path,
                    data_path=data_path,
                    project_path=project_path,
                    args=args,
                )

        deltas = {
            f"delta_{metric}": results["candidate"][metric] - results["baseline"][metric]
            for metric in COMPARISON_METRICS
        }
        log_metrics_if_active(deltas)

    print_comparison(results["baseline"], results["candidate"])
    print()
    verdict = (
        "IL NOSTRO MODELLO MIGLIORA"
        if deltas["delta_map50_95"] > 0
        else "IL NOSTRO MODELLO NON MIGLIORA"
    )
    print(f"Esito principale (mAP50-95): {verdict}")


if __name__ == "__main__":
    main()
