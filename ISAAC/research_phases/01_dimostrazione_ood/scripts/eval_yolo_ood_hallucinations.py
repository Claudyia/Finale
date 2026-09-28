# Script per valutare le allucinazioni 
# OOD del modello YOLO PPE usando il contesto di pose/persona.
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG") # Estensioni di immagini supportate
TARGET_NAMES = {"helmet", "vest"}
POSE_CONTEXT_NAMES = {"person"}


def iter_images(root: Path, recursive: bool = False):
    """""funzionche che itera su tutte le immagini in una directory, 
    con estensioni supportate.
    """  # noqa: D205, D210
    for ext in IMAGE_EXTS:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        yield from sorted(root.glob(pattern))


def find_label(labels_dir: Path, image_path: Path) -> Path | None:
    """""funzionche che trova il file di label corrispondente a un'immagine,
assumendo che abbiano lo stesso nome di base e estensioni supportate.
    es. img_001.jpg -> img_001.txt.""" 
    
    
    label_path = labels_dir / f"{image_path.stem}.txt"
    return label_path if label_path.exists() else None


def read_yolo_boxes(label_path: Path, image_size: tuple[int, int]) -> list[dict]:
    """""funzionche che legge le bounding boxes da un file di label YOLO."""
    width, height = image_size # dimensioni dell'immagine per convertire le coordinate relative in pixel
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0])) 
        x_center, y_center, box_w, box_h = map(float, parts[1:5])
        pixel_w = box_w * width
        pixel_h = box_h * height
        x1 = x_center * width - pixel_w / 2
        y1 = y_center * height - pixel_h / 2
        boxes.append(
            {
                "class_id": class_id,
                "x1": x1,
                "y1": y1,
                "x2": x1 + pixel_w,
                "y2": y1 + pixel_h,
            }
        )
    return boxes


def iou(box_a: dict, box_b: dict) -> float:
    """""funzionche che calcola l'Intersection over Union (IoU) tra due bounding boxes 
    IoU è una metrica di sovrapposizione che misura il rapporto tra l'area di sovrapposizione e 
    l'area dell'unione dei due bounding boxes.
    Area di sovrapposizione: area della regione comune tra le due boxes.
    Area di unione: area totale coperta da entrambe le boxes, calcolata come somma delle aree individuali meno l'area di sovrapposizione."""
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detection_rows(
    model: YOLO,
    image_path: Path,
    conf: float,
    imgsz: int,
    target_names: set[str] | None = None,
) -> list[dict]:
    
    
    """"funzionche che esegue la predizione del modello su un'immagine e restituisce una 
    lista di dizionari con i dettagli delle detections.
Ogni dizionario contiene:
- class_id: ID della classe rilevata
- class_name: nome della classe rilevata
- confidence: confidenza della rilevazione
- x1, y1, x2, y2: coordinate della bounding box in pixel
Il parametro target_names permette di filtrare le detections restituite, mantenendo solo quelle"""
    result = model.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0]
    rows = []
    if result.boxes is None:
        return rows
    for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        class_id_int = int(class_id.item())
        class_name = model.names[class_id_int]
        if target_names is not None and class_name not in target_names:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.cpu().tolist()]
        rows.append(
            {
                "class_id": class_id_int,
                "class_name": class_name,
                "confidence": float(score.item()),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return rows


def has_pose_context(det: dict, pose_rows: list[dict]) -> bool:
    """"funzionche che  verifica se una detection PPE ha un contesto di pose/persona associato.
Il contesto è definito come la presenza di almeno una box di persona che contiene 
il centro della detection PPE o che ha un IoU maggiore di 0 con essa
serve per determinare se la detection è plausibile in base al contesto visivo."""
    det_center_x = (det["x1"] + det["x2"]) / 2
    det_center_y = (det["y1"] + det["y2"]) / 2

    for pose in pose_rows:
        center_inside_pose = (
            pose["x1"] <= det_center_x <= pose["x2"]
            and pose["y1"] <= det_center_y <= pose["y2"]
        )
        if center_inside_pose or iou(det, pose) > 0:
            return True
    return False


def source_group(path: Path, root: Path) -> str:
    """"funzionche che stabilisce il gruppo di origine di un'immagine OOD 
     basato sulla sua posizione relativa alla directory radice."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return "unknown"
    return rel.parts[0] if len(rel.parts) > 1 else root.name


def eval_ood(model: YOLO, pose_model: YOLO | None, args) -> list[dict]:
    """"funzionche che  valuta le allucinazioni OOD su un set di immagini OOD, 
    restituendo una lista di dizionari con i dettagli delle detections."""
    rows = []
    image_paths = list(iter_images(Path(args.ood_images_dir), recursive=True))
    if args.ood_limit:
        image_paths = image_paths[: args.ood_limit]

    for image_path in image_paths: 
        pose_rows = []
        if pose_model is not None:
            pose_rows = detection_rows(
                pose_model,
                image_path,
                conf=args.pose_conf,
                imgsz=args.pose_imgsz,
                target_names=POSE_CONTEXT_NAMES,
            )

        detections = detection_rows(
            model,
            image_path,
            conf=args.conf,
            imgsz=args.imgsz,
            target_names=TARGET_NAMES,
        )
        group = source_group(image_path, Path(args.ood_images_dir)) # determina il gruppo di origine dell'immagine OOD, serve per analizzare se alcune fonti contribuiscono più di altre alle allucinazioni.
        for det in detections:
            has_context = has_pose_context(det, pose_rows) if pose_model is not None else ""
            if args.require_pose_context and not has_context:
                continue
            rows.append(
                {
                    "split": "ood",
                    "group": group,
                    "image": str(image_path),
                    "class_name": det["class_name"],
                    "confidence": f"{det['confidence']:.6f}",
                    "pose_count": len(pose_rows) if pose_model is not None else "",
                    "has_pose_context": has_context,
                    "outcome": "hallucination",
                    "matched_iou": "",
                    "x1": f"{det['x1']:.2f}",
                    "y1": f"{det['y1']:.2f}",
                    "x2": f"{det['x2']:.2f}",
                    "y2": f"{det['y2']:.2f}",
                }
            )
    return rows


def eval_id(model: YOLO, pose_model: YOLO | None, args) -> list[dict]:
    """"funzionche che valuta le detections su un set di immagini ID,
    confrontandole con le ground truth
    diverso da eval_ood perché qui possiamo calcolare true positive e false positive,
    mentre in eval_ood tutte le detections sono considerate allucinazioni."""
    rows = []
    images_dir = Path(args.id_images_dir)
    labels_dir = Path(args.id_labels_dir)
    image_paths = list(iter_images(images_dir, recursive=False))
    if args.id_limit:
        image_paths = image_paths[: args.id_limit]

    for image_path in image_paths:
        label_path = find_label(labels_dir, image_path)
        if label_path is None:
            continue
        image_size = Image.open(image_path).size
        gt_boxes = read_yolo_boxes(label_path, image_size)
        pose_rows = []
        if pose_model is not None:
            pose_rows = detection_rows(
                pose_model,
                image_path,
                conf=args.pose_conf,
                imgsz=args.pose_imgsz,
                target_names=POSE_CONTEXT_NAMES,
            )

        detections = detection_rows(
            model,
            image_path,
            conf=args.conf,
            imgsz=args.imgsz,
            target_names=TARGET_NAMES,
        )
        for det in detections:
            has_context = has_pose_context(det, pose_rows) if pose_model is not None else ""
            if args.require_pose_context and not has_context:
                continue
            best_iou = 0.0
            for gt in gt_boxes:
                if gt["class_id"] != det["class_id"]:
                    continue
                best_iou = max(best_iou, iou(det, gt))
            outcome = "id_true_positive" if best_iou >= args.iou_threshold else "id_false_positive"
            rows.append(
                {
                    "split": "id",
                    "group": "id_val",
                    "image": str(image_path),
                    "class_name": det["class_name"],
                    "confidence": f"{det['confidence']:.6f}",
                    "pose_count": len(pose_rows) if pose_model is not None else "",
                    "has_pose_context": has_context,
                    "outcome": outcome,
                    "matched_iou": f"{best_iou:.6f}",
                    "x1": f"{det['x1']:.2f}",
                    "y1": f"{det['y1']:.2f}",
                    "x2": f"{det['x2']:.2f}",
                    "y2": f"{det['y2']:.2f}",
                }
            )
    return rows


def print_summary(rows: list[dict]) -> None:
    """"funzionche che stampa un riepilogo delle detections, con conteggi e statistiche di confidenza per ID e OOD."""
    by_outcome = Counter(row["outcome"] for row in rows)
    by_split = Counter(row["split"] for row in rows)
    by_class = Counter(row["class_name"] for row in rows)
    by_group = Counter(row["group"] for row in rows if row["split"] == "ood")

    print("Detections by split:", dict(by_split))
    print("Detections by outcome:", dict(by_outcome))
    print("Detections by class:", dict(by_class))
    if by_group:
        print("OOD hallucinations by group:")
        for group, count in by_group.most_common():
            print(f"  {group}: {count}")

    confidences_by_outcome = defaultdict(list) # dizionario che raggruppa le confidenze delle detections per ogni tipo di outcome (es. id_true_positive, id_false_positive, hallucination)
    for row in rows:
        confidences_by_outcome[row["outcome"]].append(float(row["confidence"]))
    for outcome, values in sorted(confidences_by_outcome.items()):
        mean_conf = sum(values) / len(values)
        print(f"{outcome}: n={len(values)}, mean_conf={mean_conf:.3f}, max_conf={max(values):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PPE hallucinations on semantic OOD images using pose/person context by default."
    )
    parser.add_argument("--model", default="models/yolov8n-ppe_run_1_classes_1_2.pt")
    parser.add_argument("--id-images-dir", default="scripts/finetunig/dataset/finetune_dataset/images/val")
    parser.add_argument("--id-labels-dir", default="scripts/finetunig/dataset/finetune_dataset/labels/val")
    parser.add_argument("--ood-images-dir", default="isaac_odd_dataset")
    parser.add_argument("--output-dir", default="ood_hallucination_eval")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pose-model", default="models/yolov8n-pose.pt")
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument(
        "--require-pose-context",
        action="store_true",
        default=True,
        help="Keep only PPE detections whose center is inside a pose/person box or overlaps it.",
    )
    parser.add_argument(
        "--no-pose-context",
        dest="require_pose_context",
        action="store_false",
        help="Debug only: evaluate the PPE detector without requiring a pose/person context.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--id-limit", type=int, default=0)
    parser.add_argument("--ood-limit", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(args.model)
    pose_model = YOLO(args.pose_model) if args.pose_model else None
    rows = []
    rows.extend(eval_id(model, pose_model, args))
    rows.extend(eval_ood(model, pose_model, args))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "yolo_ood_hallucinations.csv"
    fieldnames = [
        "split",
        "group",
        "image",
        "class_name",
        "confidence",
        "pose_count",
        "has_pose_context",
        "outcome",
        "matched_iou",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
