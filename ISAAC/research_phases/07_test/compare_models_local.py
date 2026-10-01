from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


BASELINE_MODEL = Path("yolov8n_run_1.2_classes_1_2.pt")
CANDIDATE_MODEL = Path("ppe_run12_yoloauto_fold2_best.pt")
DATASET_YAML = Path("research_phases/dataset_test_isaac/data.yaml")
OUTPUT_PROJECT = Path("research_phases/07_test/comparison_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confronta due modelli YOLO sullo stesso dataset senza usare MLflow."
        )
    )
    parser.add_argument("--baseline", default=str(BASELINE_MODEL))
    parser.add_argument("--candidate", default=str(CANDIDATE_MODEL))
    parser.add_argument("--data", default=str(DATASET_YAML))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="", help="Esempi: cpu, mps, 0.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default=str(OUTPUT_PROJECT))
    return parser.parse_args()


def validate_path(path: str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} non trovato: {resolved}")
    return resolved


def extract_metrics(result: Any) -> dict[str, float]:
    box = result.box
    precision = np.asarray(getattr(box, "p", []), dtype=float)
    recall = np.asarray(getattr(box, "r", []), dtype=float)
    f1 = np.asarray(getattr(box, "f1", []), dtype=float)
    speed = getattr(result, "speed", {}) or {}

    return {
        "precision": float(np.mean(precision)) if precision.size else 0.0,
        "recall": float(np.mean(recall)) if recall.size else 0.0,
        "f1": float(np.mean(f1)) if f1.size else 0.0,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "inference_ms_per_image": float(speed.get("inference", 0.0)),
    }


def evaluate_model(
    label: str,
    model_path: Path,
    data_path: Path,
    args: argparse.Namespace,
    project_path: Path,
) -> dict[str, Any]:
    validation_args: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "workers": args.workers,
        "project": str(project_path),
        "name": label,
        "plots": True,
        "save_json": True,
    }
    if args.device:
        validation_args["device"] = args.device

    print()
    print(f"Valutazione {label}: {model_path}")
    result = YOLO(str(model_path)).val(**validation_args)

    return {
        "model": str(model_path),
        "output_directory": str(Path(result.save_dir)),
        "metrics": extract_metrics(result),
    }


def compare_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, dict[str, float | bool]]:
    comparison: dict[str, dict[str, float | bool]] = {}

    for metric, baseline_value in baseline.items():
        candidate_value = candidate[metric]
        delta = candidate_value - baseline_value
        lower_is_better = metric == "inference_ms_per_image"
        improved = delta < 0 if lower_is_better else delta > 0
        delta_percent = (
            (delta / baseline_value) * 100 if baseline_value != 0 else 0.0
        )
        comparison[metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
            "delta_percent": delta_percent,
            "improved": improved,
        }

    return comparison


def print_comparison(comparison: dict[str, dict[str, float | bool]]) -> None:
    print()
    print("CONFRONTO FINALE")
    print(f"{'Metrica':<24} {'Iniziale':>12} {'Mio':>12} {'Delta':>12} Esito")
    print("-" * 78)

    for metric, values in comparison.items():
        verdict = "MIGLIORA" if values["improved"] else "PEGGIORA/UGUALE"
        print(
            f"{metric:<24} "
            f"{values['baseline']:>12.4f} "
            f"{values['candidate']:>12.4f} "
            f"{values['delta']:>+12.4f} "
            f"{verdict}"
        )


def main() -> None:
    args = parse_args()
    baseline_path = validate_path(args.baseline, "Modello iniziale")
    candidate_path = validate_path(args.candidate, "Modello candidato")
    data_path = validate_path(args.data, "Dataset YAML")
    project_path = Path(args.project).expanduser().resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    print(f"Dataset comune: {data_path}")
    print(f"Split: {args.split}")
    print("MLflow: non utilizzato")

    baseline = evaluate_model(
        "baseline",
        baseline_path,
        data_path,
        args,
        project_path,
    )
    candidate = evaluate_model(
        "candidate",
        candidate_path,
        data_path,
        args,
        project_path,
    )
    comparison = compare_metrics(
        baseline["metrics"],
        candidate["metrics"],
    )

    report = {
        "dataset": str(data_path),
        "split": args.split,
        "configuration": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "device": args.device or "automatic",
            "workers": args.workers,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "candidate_improves_map50_95": comparison["map50_95"]["improved"],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = project_path / f"comparison_{timestamp}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print_comparison(comparison)
    print()
    print(
        "Esito principale (mAP50-95): "
        + (
            "IL MIO MODELLO MIGLIORA"
            if report["candidate_improves_map50_95"]
            else "IL MIO MODELLO NON MIGLIORA"
        )
    )
    print(f"Report locale: {report_path}")


if __name__ == "__main__":
    main()
