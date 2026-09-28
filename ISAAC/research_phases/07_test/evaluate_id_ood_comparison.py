from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.entities import ViewType
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_utils import (  # noqa: E402
    init_mlflow,
    log_artifact_path,
    safe_log_metrics,
    start_mlflow_run,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_CLASS_NAMES = {"helmet", "vest"}
POSE_CLASS_NAMES = {"person"}

EXPERIMENT_NAME = "claudia_ood_confronto"
BASELINE_RUN_NAME = "baseline_yolov8n_ppe_run1"
CANDIDATE_RUN_NAME = "run12_fold2_best"

BASELINE_MODEL = Path("models/yolov8n-ppe_run_1_classes_1_2.pt")
CANDIDATE_MODEL = Path("ppe_run12_yoloauto_fold2_best.pt")
ID_DATASET = Path("research_phases/dataset_test_isaac/data.yaml")
POSE_MODEL = Path("models/yolov8n-pose.pt")
OOD_IMAGES = Path(
    "/Users/claudia/Desktop/isac_odd/isaac_odd/"
    "isaac_odd_dataset/ood_holdout_valid/images"
)
OUTPUT_DIRECTORY = Path("research_phases/07_test/id_ood_comparison_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confronta ex novo baseline e modello candidato su dataset ID e OOD, "
            "creando due run nello stesso experiment MLflow."
        )
    )
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--baseline", default=str(BASELINE_MODEL))
    parser.add_argument("--candidate", default=str(CANDIDATE_MODEL))
    parser.add_argument("--id-data", default=str(ID_DATASET))
    parser.add_argument("--ood-images", default=str(OOD_IMAGES))
    parser.add_argument("--pose-model", default=str(POSE_MODEL))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIRECTORY))

    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--id-conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="", help="Esempi: cpu, mps, 0.")

    parser.add_argument("--ood-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--ood-limit", type=int, default=0)
    parser.add_argument(
        "--no-pose-context",
        action="store_true",
        help="Disabilita il filtro persona/pose (solo per debug).",
    )
    parser.add_argument(
        "--allow-existing-experiment",
        action="store_true",
        help="Permette di aggiungere run a un experiment che contiene già run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verifica input e configurazione senza creare experiment o run.",
    )
    return parser.parse_args()


def require_file(path: str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} non trovato: {resolved}")
    return resolved


def require_directory(path: str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} non trovata: {resolved}")
    return resolved


def iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def partition_readable_images(
    image_paths: list[Path],
) -> tuple[list[Path], list[Path]]:
    """Separa le immagini leggibili da quelle corrotte/non supportate."""

    readable: list[Path] = []
    skipped: list[Path] = []
    for image_path in image_paths:
        if cv2.imread(str(image_path)) is None:
            skipped.append(image_path)
        else:
            readable.append(image_path)
    return readable, skipped


def ensure_experiment_is_clean(experiment_name: str, allow_existing: bool) -> str:
    experiment_id = init_mlflow(experiment_name)
    existing_runs = MlflowClient().search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    if existing_runs and not allow_existing:
        raise RuntimeError(
            f"L'experiment '{experiment_name}' contiene già almeno una run. "
            "Eliminalo, scegli un altro nome oppure usa "
            "--allow-existing-experiment consapevolmente."
        )
    return experiment_id


def model_names(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def result_detections(
    model: YOLO,
    image_path: Path,
    confidence: float,
    image_size: int,
    device: str,
    accepted_names: set[str],
) -> list[dict[str, Any]]:
    predict_args: dict[str, Any] = {
        "source": str(image_path),
        "conf": confidence,
        "imgsz": image_size,
        "verbose": False,
    }
    if device:
        predict_args["device"] = device

    results = model.predict(**predict_args)
    if not results:
        return []
    result = results[0]
    if result.boxes is None:
        return []

    names = model_names(model)
    detections: list[dict[str, Any]] = []
    for box, score, class_id in zip(
        result.boxes.xyxy,
        result.boxes.conf,
        result.boxes.cls,
    ):
        class_index = int(class_id.item())
        class_name = names[class_index]
        if class_name not in accepted_names:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.cpu().tolist()]
        detections.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "confidence": float(score.item()),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


def intersection_over_union(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(
        0.0, box_a["y2"] - box_a["y1"]
    )
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(
        0.0, box_b["y2"] - box_b["y1"]
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def has_pose_context(
    detection: dict[str, Any],
    person_detections: list[dict[str, Any]],
) -> bool:
    center_x = (detection["x1"] + detection["x2"]) / 2
    center_y = (detection["y1"] + detection["y2"]) / 2

    for person in person_detections:
        center_inside = (
            person["x1"] <= center_x <= person["x2"]
            and person["y1"] <= center_y <= person["y2"]
        )
        if center_inside or intersection_over_union(detection, person) > 0:
            return True
    return False


def mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def extract_id_metrics(metrics: Any) -> tuple[dict[str, float], dict[str, Any]]:
    box = metrics.box
    precision = np.asarray(getattr(box, "p", []), dtype=float)
    recall = np.asarray(getattr(box, "r", []), dtype=float)
    f1 = np.asarray(getattr(box, "f1", []), dtype=float)
    ap50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    maps = np.asarray(getattr(box, "maps", []), dtype=float)
    names = getattr(metrics, "names", {})

    mlflow_metrics: dict[str, float] = {
        "id_precision": mean_or_zero(precision),
        "id_recall": mean_or_zero(recall),
        "id_f1": mean_or_zero(f1),
        "id_map50": float(box.map50),
        "id_map50_95": float(box.map),
    }
    per_class: list[dict[str, Any]] = []
    class_count = max(len(precision), len(recall), len(f1), len(ap50), len(maps))
    for index in range(class_count):
        class_name = (
            str(names.get(index, names.get(str(index), index)))
            if isinstance(names, dict)
            else str(names[index])
        )
        class_key = class_name.lower().replace(" ", "_")
        values = {
            "class_id": index,
            "class_name": class_name,
            "precision": float(precision[index]) if index < len(precision) else 0.0,
            "recall": float(recall[index]) if index < len(recall) else 0.0,
            "f1": float(f1[index]) if index < len(f1) else 0.0,
            "map50": float(ap50[index]) if index < len(ap50) else 0.0,
            "map50_95": float(maps[index]) if index < len(maps) else 0.0,
        }
        per_class.append(values)
        for metric_name in ("precision", "recall", "f1", "map50", "map50_95"):
            mlflow_metrics[f"id_{class_key}_{metric_name}"] = float(
                values[metric_name]
            )

    report = {
        "overall": {
            "precision": mlflow_metrics["id_precision"],
            "recall": mlflow_metrics["id_recall"],
            "f1": mlflow_metrics["id_f1"],
            "map50": mlflow_metrics["id_map50"],
            "map50_95": mlflow_metrics["id_map50_95"],
        },
        "per_class": per_class,
        "speed_ms_per_image": getattr(metrics, "speed", {}),
    }
    return mlflow_metrics, report


def evaluate_id(
    model: YOLO,
    data_path: Path,
    run_name: str,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any], Path]:
    validation_args: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.id_conf,
        "iou": args.iou,
        "workers": args.workers,
        "project": str(output_root / "id_validation"),
        "name": run_name,
        "exist_ok": True,
        "plots": True,
        "save_json": False,
    }
    if args.device:
        validation_args["device"] = args.device

    metrics = model.val(**validation_args)
    mlflow_metrics, report = extract_id_metrics(metrics)
    return mlflow_metrics, report, Path(metrics.save_dir)


def evaluate_ood(
    model: YOLO,
    pose_model: YOLO,
    image_paths: list[Path],
    args: argparse.Namespace,
    csv_path: Path,
    skipped_image_count: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    images_with_raw_detection: set[str] = set()
    images_with_hallucination: set[str] = set()
    raw_detection_count = 0

    for position, image_path in enumerate(image_paths, start=1):
        print(f"OOD {position}/{len(image_paths)}: {image_path.name}", end="\r")
        person_detections = result_detections(
            pose_model,
            image_path,
            args.pose_conf,
            args.imgsz,
            args.device,
            POSE_CLASS_NAMES,
        )
        ppe_detections = result_detections(
            model,
            image_path,
            args.ood_conf,
            args.imgsz,
            args.device,
            TARGET_CLASS_NAMES,
        )
        raw_detection_count += len(ppe_detections)
        if ppe_detections:
            images_with_raw_detection.add(str(image_path))

        for detection in ppe_detections:
            pose_context = has_pose_context(detection, person_detections)
            if not args.no_pose_context and not pose_context:
                continue
            images_with_hallucination.add(str(image_path))
            rows.append(
                {
                    "image": str(image_path),
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "pose_count": len(person_detections),
                    "has_pose_context": pose_context,
                    "x1": detection["x1"],
                    "y1": detection["y1"],
                    "x2": detection["x2"],
                    "y2": detection["y2"],
                }
            )
    print()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "class_name",
        "confidence",
        "pose_count",
        "has_pose_context",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_images = len(image_paths)
    confidences = [float(row["confidence"]) for row in rows]
    class_counts = Counter(str(row["class_name"]) for row in rows)
    mlflow_metrics = {
        "ood_images_total": float(total_images),
        "ood_images_skipped_unreadable": float(skipped_image_count),
        "ood_raw_detections_total": float(raw_detection_count),
        "ood_images_with_raw_detection": float(len(images_with_raw_detection)),
        "ood_hallucinations_total": float(len(rows)),
        "ood_images_with_hallucination": float(len(images_with_hallucination)),
        "ood_image_hallucination_rate": (
            len(images_with_hallucination) / total_images if total_images else 0.0
        ),
        "ood_hallucinations_per_image": (
            len(rows) / total_images if total_images else 0.0
        ),
        "ood_mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "ood_max_confidence": max(confidences, default=0.0),
        "ood_helmet_hallucinations": float(class_counts.get("helmet", 0)),
        "ood_vest_hallucinations": float(class_counts.get("vest", 0)),
    }
    report = {
        "metrics": mlflow_metrics,
        "class_counts": dict(class_counts),
        "csv": str(csv_path),
    }
    return mlflow_metrics, report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def comparison_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    baseline_hallucinations = baseline["ood_hallucinations_total"]
    candidate_hallucinations = candidate["ood_hallucinations_total"]
    reduction = (
        (baseline_hallucinations - candidate_hallucinations)
        / baseline_hallucinations
        * 100
        if baseline_hallucinations
        else 0.0
    )
    return {
        "comparison_id_map50_delta": (
            candidate["id_map50"] - baseline["id_map50"]
        ),
        "comparison_id_map50_95_delta": (
            candidate["id_map50_95"] - baseline["id_map50_95"]
        ),
        "comparison_id_recall_delta": (
            candidate["id_recall"] - baseline["id_recall"]
        ),
        "comparison_ood_hallucination_delta": (
            candidate_hallucinations - baseline_hallucinations
        ),
        "comparison_ood_hallucination_reduction_percent": reduction,
    }


def main() -> None:
    args = parse_args()
    baseline_path = require_file(args.baseline, "Modello baseline")
    candidate_path = require_file(args.candidate, "Modello candidato")
    id_data_path = require_file(args.id_data, "Dataset ID")
    pose_model_path = require_file(args.pose_model, "Modello pose")
    ood_directory = require_directory(args.ood_images, "Cartella immagini OOD")
    output_root = Path(args.output_dir).expanduser().resolve()

    discovered_image_paths = iter_images(ood_directory)
    image_paths, skipped_image_paths = partition_readable_images(
        discovered_image_paths
    )
    if args.ood_limit:
        image_paths = image_paths[: args.ood_limit]
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine OOD trovata in: {ood_directory}")

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {args.experiment}")
    print(f"Dataset ID: {id_data_path}")
    print(
        f"Dataset OOD: {ood_directory} "
        f"({len(image_paths)} leggibili, {len(skipped_image_paths)} escluse)"
    )
    for skipped_path in skipped_image_paths:
        print(f"Immagine OOD esclusa: {skipped_path}")
    print(f"Contesto pose: {not args.no_pose_context}")
    print(f"Baseline: {baseline_path}")
    print(f"Candidato: {candidate_path}")

    if args.dry_run:
        print("Dry-run completato: nessun experiment o run è stato creato.")
        return

    ensure_experiment_is_clean(args.experiment, args.allow_existing_experiment)

    pose_model = YOLO(str(pose_model_path))
    configurations = [
        (BASELINE_RUN_NAME, baseline_path, "baseline"),
        (CANDIDATE_RUN_NAME, candidate_path, "candidate"),
    ]
    completed: dict[str, dict[str, Any]] = {}

    for run_name, model_path, role in configurations:
        parameters = {
            "role": role,
            "model": str(model_path),
            "id_dataset": str(id_data_path),
            "ood_dataset": str(ood_directory),
            "ood_image_count": len(image_paths),
            "ood_skipped_unreadable": len(skipped_image_paths),
            "pose_model": str(pose_model_path),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "id_conf": args.id_conf,
            "iou": args.iou,
            "ood_conf": args.ood_conf,
            "pose_conf": args.pose_conf,
            "require_pose_context": not args.no_pose_context,
            "device": args.device or "automatic",
            "workers": args.workers,
        }

        with start_mlflow_run(
            experiment_name=args.experiment,
            run_name=run_name,
            params=parameters,
            tags={
                "evaluation": "fresh_id_ood_comparison",
                "model_role": role,
                "task": "ppe-object-detection",
            },
        ):
            print()
            print(f"Run: {run_name}")
            model = YOLO(str(model_path))
            id_metrics, id_report, id_output = evaluate_id(
                model,
                id_data_path,
                run_name,
                output_root,
                args,
            )
            safe_log_metrics(id_metrics)
            log_artifact_path(id_output, artifact_path="id_validation")

            ood_csv = output_root / run_name / "ood_hallucinations.csv"
            ood_metrics, ood_report = evaluate_ood(
                model,
                pose_model,
                image_paths,
                args,
                ood_csv,
                len(skipped_image_paths),
            )
            safe_log_metrics(ood_metrics)
            all_metrics = {**id_metrics, **ood_metrics}
            run_report = {
                "run_name": run_name,
                "role": role,
                "model": str(model_path),
                "id": id_report,
                "ood": ood_report,
            }
            report_path = output_root / run_name / "evaluation_report.json"
            write_json(report_path, run_report)

            if role == "candidate" and "baseline" in completed:
                deltas = comparison_metrics(completed["baseline"]["metrics"], all_metrics)
                all_metrics.update(deltas)
                comparison_report = {
                    "baseline": completed["baseline"],
                    "candidate": {"report": run_report, "metrics": all_metrics},
                    "comparison": deltas,
                }
                comparison_path = output_root / "comparison_report.json"
                write_json(comparison_path, comparison_report)
                safe_log_metrics(deltas)
                log_artifact_path(comparison_path, artifact_path="comparison")

            log_artifact_path(ood_csv, artifact_path="ood_evaluation")
            log_artifact_path(report_path, artifact_path="reports")
            completed[role] = {"report": run_report, "metrics": all_metrics}

    deltas = comparison_metrics(
        completed["baseline"]["metrics"],
        completed["candidate"]["metrics"],
    )
    print()
    print("CONFRONTO COMPLETATO")
    print(
        "Riduzione hallucination OOD: "
        f"{deltas['comparison_ood_hallucination_reduction_percent']:.2f}%"
    )
    print(f"Report: {output_root / 'comparison_report.json'}")


if __name__ == "__main__":
    main()
