#codice per creare un dataset YOLO con immagini OOD hard
# negative per ridurre le allucinazioni di YOLO su oggetti OOD semantici. 
# Copia il dataset di base e aggiunge immagini OOD al training split con file di 
# etichette YOLO vuoti.
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def copy_tree_files(src: Path, dst: Path) -> int: 
    """copia ricorsivamente i file mantenendo la struttura delle cartelle."""
    count = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src) # percorso relativo alla cartella di origine
        out = dst / rel # percorso di destinazione mantenendo la struttura
        out.parent.mkdir(parents=True, exist_ok=True) # crea le cartelle di destinazione se non esistono
        shutil.copy2(path, out)
        count += 1
    return count


def read_hallucination_images(report_path: Path) -> list[Path]:
    """"Legge il report CSV e restituisce una lista di percorsi di
    immagini che sono state classificate come allucinazioni OOD.
    Il report CSV dovrebbe avere colonne: split, outcome, image."""
    images = []
    seen = set()
    with report_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != "ood":
                continue
            if row.get("outcome") != "hallucination":
                continue
            image = Path(row["image"])
            if image in seen:
                continue
            seen.add(image)
            images.append(image)
    return images


def all_images(root: Path) -> list[Path]: 
    """Restituisce una lista di tutti i percorsi di immagini sotto la cartella root con estensioni supportate."""
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in IMAGE_EXTS)


def copy_negative_images(images: list[Path], out_images_dir: Path, out_labels_dir: Path, max_images: int) -> int:
    """Copia le immagini negative OOD nella cartella di output e crea file di etichette
vuoti corrispondenti."""


    count = 0
    for image_path in images:
        if max_images and count >= max_images:
            break
        if not image_path.exists():
            continue
        safe_stem = image_path.stem.replace(" ", "_").replace("/", "_")
        out_name = f"oodneg_{count:05d}_{safe_stem}{image_path.suffix.lower()}"
        out_image = out_images_dir / out_name
        out_label = out_labels_dir / f"{Path(out_name).stem}.txt"
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_image)
        out_label.write_text("")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO dataset with OOD hard negatives.")
    parser.add_argument("--base-dataset", default="scripts/finetunig/dataset/finetune_dataset")
    parser.add_argument("--hallucination-report", default="ood_hallucination_eval/yolo_ood_hallucinations.csv")
    parser.add_argument("--ood-images-dir", default="isaac_odd_dataset")
    parser.add_argument("--output-dataset", default="scripts/finetunig/dataset/finetune_dataset_ood_hardneg")
    parser.add_argument("--max-negatives", type=int, default=300)
    parser.add_argument("--use-all-ood", action="store_true", help="Use all OOD images instead of only hallucination images from the report.")
    args = parser.parse_args()

    base = Path(args.base_dataset)
    output = Path(args.output_dataset)
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")

    for split in ("train", "val"):
        copy_tree_files(base / "images" / split, output / "images" / split)
        copy_tree_files(base / "labels" / split, output / "labels" / split)

    if args.use_all_ood:
        negative_images = all_images(Path(args.ood_images_dir))
    else:
        negative_images = read_hallucination_images(Path(args.hallucination_report))

    added = copy_negative_images(
        images=negative_images,
        out_images_dir=output / "images" / "train",
        out_labels_dir=output / "labels" / "train",
        max_images=args.max_negatives,
    )

    (output / "data.yaml").write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: helmet\n"
        "  1: vest\n"
    )

    (output / "README.md").write_text(
        "# PPE Dataset With OOD Hard Negatives\n\n"
        "This dataset copies the original PPE fine-tuning dataset and adds OOD images to the training split with empty YOLO label files.\n\n"
        "Purpose: reduce YOLO hallucinations on semantic OOD objects by explicit hard-negative exposure.\n\n"
        f"Base dataset: `{base}`\n"
        f"Added OOD negative images: `{added}`\n"
        f"Source mode: `{'all_ood' if args.use_all_ood else 'hallucination_report'}`\n"
    )

    print(f"Created dataset: {output}")
    print(f"Added OOD hard negatives: {added}")
    print(f"Data YAML: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
