from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO
import mlflow
from scripts.mlflow_utils import init_mlflow, safe_log_params, log_artifact_path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def convert_numpy_scalars(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: convert_numpy_scalars(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(convert_numpy_scalars(value) for value in obj)
    return obj


def collect_images(images_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(images_dir.rglob(f"*{ext}"))
    return sorted(images)


def create_split_txts(images_dir: Path, output_dir: Path, seed: int, split_ratio: float) -> tuple[Path, Path]:
    rng = random.Random(seed)
    images = [image.resolve() for image in collect_images(images_dir)]
    if not images:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    rng.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    train_txt.write_text("\n".join(str(image) for image in train_images) + "\n")
    val_txt.write_text("\n".join(str(image) for image in val_images) + "\n")
    return train_txt, val_txt


def generate_virtual_data_yaml(original_yaml_path: Path, train_txt: Path, val_txt: Path, output_yaml_path: Path) -> None:
    original_yaml = yaml.safe_load(original_yaml_path.read_text())
    names = original_yaml["names"]
    virtual_yaml = {
        "path": ".",
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "names": names,
        "nc": original_yaml.get("nc", len(names)),
    }
    output_yaml_path.write_text(yaml.safe_dump(virtual_yaml, sort_keys=False))


def save_metrics(metrics: Any, output_json_path: Path) -> None:
    metric_dict = {
        "metrics/mAP50": metrics.box.map50,
        "metrics/mAP50-95": metrics.box.map,
        "metrics/precision": float(np.mean(metrics.box.p)),
        "metrics/recall": float(np.mean(metrics.box.r)),
        "metrics/f1": float(np.mean(metrics.box.f1)),
        "class_map50": getattr(metrics.box, "ap50", None),
        "class_map5095": metrics.box.maps,
        "class_f1": getattr(metrics.box, "f1", None),
        "class_precision": getattr(metrics.box, "p", None),
        "class_recall": getattr(metrics.box, "r", None),
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(convert_numpy_scalars(metric_dict), indent=2))
    print(f"Metrics saved to: {output_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: train an initial YOLO checkpoint.")
    parser.add_argument("--data", default="scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost_helmetboost/data.yaml")
    parser.add_argument("--model", default="yolov8n_run_1.2_classes_1_2.pt")
    parser.add_argument("--project", default="finetune_runs/phase_1_initial")
    parser.add_argument("--name", default="ppe_ood_hardneg_vestboost_helmetboost_initial")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--lrf", type=float, default=1.0)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--perspective", type=float, default=0.01)
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    dataset_dir = data_yaml.parent
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    split_dir = dataset_dir / "virtual_split"
    train_txt, val_txt = create_split_txts(images_dir, split_dir, args.seed, args.split_ratio)
    virtual_yaml_path = split_dir / "data_virtual.yaml"
    generate_virtual_data_yaml(data_yaml, train_txt, val_txt, virtual_yaml_path)

    train_kwargs = {
        "data": str(virtual_yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "workers": args.workers,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "dropout": args.dropout,
        "mosaic": args.mosaic,
        "perspective": args.perspective,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model = YOLO(args.model)
    # initialize MLflow experiment
    init_mlflow(args.project)

    with mlflow.start_run(run_name=args.name):
        # log selected training params
        safe_log_params(
            {
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "lr0": args.lr0,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "mosaic": args.mosaic,
                "perspective": args.perspective,
                "seed": args.seed,
                "model": args.model,
                "data": str(virtual_yaml_path),
            }
        )

        model.train(**train_kwargs)

        val_kwargs = {"data": str(virtual_yaml_path), "imgsz": args.imgsz, "batch": args.batch}
        if args.device:
            val_kwargs["device"] = args.device
        metrics = model.val(**val_kwargs)

        # save metrics JSON and log as artifact
        metrics_json = Path(args.project) / args.name / "validation_metrics.json"
        save_metrics(metrics, metrics_json)
        log_artifact_path(metrics_json)

        # log some scalar metrics to MLflow (if present)
        try:
            mlflow.log_metric("metrics/mAP50", float(metrics.box.map50))
        except Exception:
            pass
        try:
            mlflow.log_metric("metrics/mAP50-95", float(metrics.box.map))
        except Exception:
            pass

        # log the best model if available
        best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
        log_artifact_path(best_model_path)

    print("Initial checkpoint:")
    print(Path(args.project) / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
