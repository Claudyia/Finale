from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

# Permette di eseguire questo file direttamente dalla riga di comando e di
# importare i moduli condivisi presenti nella root del progetto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_utils import (
    log_artifact_path,
    safe_log_metrics,
    start_mlflow_run,
)


# Cambia questi path quando vuoi testare un altro modello o un altro dataset.
MODEL_PATH = Path("/Users/claudia/Desktop/ROBA ISAAC copia/ISAAC_/isaac_odd originale/yolov8n_run_1.2_classes_1_2.pt")
DATASET_YAML_PATH = Path(
    "/Users/claudia/Desktop/ROBA ISAAC copia/ISAAC_/isaac_odd originale/research_phases/dataset_test_isaac/data.yaml"
)

# Output del test.
PROJECT_PATH = Path("/Users/claudia/Desktop/ROBA ISAAC copia/ISAAC_/isaac_odd originale/research_phases/07_test/test_runs")
RUN_NAME = "ppe_run12_fold2_on_dataset"


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


def build_metrics_report(metrics: Any, args: argparse.Namespace) -> dict[str, Any]:
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
        "configuration": {
            "model": str(Path(args.model).resolve()),
            "data": str(Path(args.data).resolve()),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "device": args.device or "automatic",
        },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valuta un modello YOLO Ultralytics su un dataset YOLO."
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Percorso del modello da testare.",
    )
    parser.add_argument(
        "--data",
        default=str(DATASET_YAML_PATH),
        help="Percorso del file data.yaml del dataset di test.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Split dichiarato nel data.yaml da valutare.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="", help="Esempi: cpu, mps, 0.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default=str(PROJECT_PATH))
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument(
        "--experiment",
        default="claudia_ood_test_isaac_model",
        help="Nome dell'experiment MLflow che raggruppa le run di test.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    project_path = Path(args.project).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML non trovato: {data_path}")

    val_kwargs: dict[str, Any] = {
        "data": str(data_path.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "workers": args.workers,
        "project": str(project_path),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": True,
        "save_json": True,
    }
    if args.device:
        val_kwargs["device"] = args.device

    print(f"Modello: {model_path}")
    print(f"Dataset YAML: {data_path}")
    print(f"Split: {args.split}")

    run_params = {
        "model": str(model_path.resolve()),
        "dataset": str(data_path.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": args.device or "automatic",
        "workers": args.workers,
    }

    with start_mlflow_run(
        experiment_name=args.experiment,
        run_name=args.name,
        params=run_params,
        tags={
            "pipeline": "ultralytics-validation",
            "task": "object-detection",
        },
    ) as run:
        print(f"Experiment MLflow: {args.experiment}")
        print(f"Run MLflow: {run.info.run_id}")

        metrics = YOLO(str(model_path)).val(**val_kwargs)
        output_dir = Path(metrics.save_dir)
        report_path = output_dir / "test_metrics.json"
        report = build_metrics_report(metrics, args)
        report_path.write_text(
            json.dumps(to_json_value(report), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        safe_log_metrics(
            {
                "precision": report["overall"]["precision"],
                "recall": report["overall"]["recall"],
                "f1": report["overall"]["f1"],
                "map50": report["overall"]["mAP50"],
                "map50_95": report["overall"]["mAP50-95"],
                "fitness": report["overall"]["fitness"],
            }
        )
        log_artifact_path(output_dir, artifact_path="ultralytics-results")

        print()
        print("Test completato.")
        print(f"mAP50: {report['overall']['mAP50']:.4f}")
        print(f"mAP50-95: {report['overall']['mAP50-95']:.4f}")
        print(f"Risultati: {output_dir}")
        print(f"Report JSON: {report_path}")


if __name__ == "__main__":
    main()
