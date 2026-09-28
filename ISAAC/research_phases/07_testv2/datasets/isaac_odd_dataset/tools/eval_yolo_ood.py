#!/usr/bin/env python3
"""Run a YOLO model on ID/OOD image folders and export detection statistics.

This is a first-pass OOD sanity check for ISaAC. It does not modify the model.
It reports whether YOLO fires with high confidence on OOD images.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def summarize_result(result) -> dict[str, object]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "num_detections": 0,
            "max_conf": 0.0,
            "mean_conf": 0.0,
            "classes": "",
        }

    confs = boxes.conf.cpu().numpy().tolist()
    classes = [int(value) for value in boxes.cls.cpu().numpy().tolist()]
    names = getattr(result, "names", {}) or {}
    class_names = [str(names.get(cls, cls)) for cls in classes]
    return {
        "num_detections": len(confs),
        "max_conf": max(confs),
        "mean_conf": sum(confs) / len(confs),
        "classes": "|".join(class_names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO detection behavior on ID/OOD image folders."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--limit-id", default=500, type=int)
    parser.add_argument("--limit-ood", default=500, type=int)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it with "
            "`python3 -m pip install ultralytics` or run this script inside "
            "the ISaAC environment."
        ) from exc

    dataset_root = args.dataset_root.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    id_root = dataset_root / "id"
    ood_root = dataset_root / "ood"

    if not id_root.exists():
        raise FileNotFoundError(f"Missing ID folder: {id_root}")
    if not ood_root.exists():
        raise FileNotFoundError(f"Missing OOD folder: {ood_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    samples = [(path, "id") for path in list(iter_images(id_root))[: args.limit_id]]
    samples.extend((path, "ood") for path in list(iter_images(ood_root))[: args.limit_ood])

    model = YOLO(str(model_path))
    rows = []
    for image_path, split in samples:
        results = model.predict(
            str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )
        summary = summarize_result(results[0])
        rows.append(
            {
                "file": str(image_path),
                "split": split,
                **summary,
                "ood_suspect_high_conf": (
                    split == "ood"
                    and summary["num_detections"] > 0
                    and float(summary["max_conf"]) >= 0.5
                ),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "split",
                "num_detections",
                "max_conf",
                "mean_conf",
                "classes",
                "ood_suspect_high_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    id_rows = [row for row in rows if row["split"] == "id"]
    ood_rows = [row for row in rows if row["split"] == "ood"]
    ood_high = [row for row in ood_rows if row["ood_suspect_high_conf"]]

    print(f"id_images: {len(id_rows)}")
    print(f"ood_images: {len(ood_rows)}")
    print(
        "id_detection_rate: "
        f"{sum(row['num_detections'] > 0 for row in id_rows) / max(len(id_rows), 1):.3f}"
    )
    print(
        "ood_detection_rate: "
        f"{sum(row['num_detections'] > 0 for row in ood_rows) / max(len(ood_rows), 1):.3f}"
    )
    print(f"ood_high_conf_suspects: {len(ood_high)}")
    print(f"csv: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
