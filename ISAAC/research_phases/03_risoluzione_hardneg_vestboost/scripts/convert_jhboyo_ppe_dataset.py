#codice per convertire il dataset jhboyo/ppe-dataset in un formato compatibile con YOLOv8,
# mantenendo solo le classi di interesse (helmet e vest) e rinominando i file in
# modo uniforme. Il dataset convertito sarà salvato in una nuova directory con la stessa struttura
# di immagini e label, e includerà un file data.yaml con le classi e gli split corretti. 
 
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG") #2: 1 indica che la classe "vest" del dataset originale viene mappata alla classe "1 vest" nel dataset convertito, mentre la classe "helmet" rimane "0 helmet" e la classe "head" viene scartata.
CLASS_MAP = {
    0: 0,  # helmet -> helmet
    2: 1,  # vest -> vest
}
CSV_CLASS_MAP = {
    "helmet": 0,
    "hardhat": 0,
    "hard hat": 0,
    "safety helmet": 0,
    "safety vest": 1,
    "vest": 1,
}


def find_image(images_dir: Path, stem: str) -> Path | None:
    """funzionche che cerca un'immagine in una directory specificata"""
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def split_source(raw_dir: Path, split: str) -> Path | None:
    """funzionche che trova la directory corretta per uno split specifico, gestendo possibili nomi diversi come "val", "valid" o "validation" per lo split di validazione. Restituisce il percorso della directory dello split trovato o None se non esiste."""
    candidates = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
    }
    for name in candidates[split]:
        path = raw_dir / name
        if not path.exists():
            continue
        nested = path / name
        if nested.exists():
            return nested
        if name == "valid" and (path / "valid").exists():
            return path / "valid"
        if path.exists():
            return path
    return None


def convert_label(label_path: Path, require_vest: bool) -> list[str]:
    """funzionche che cerca le annotazioni nel file di label, 
    converte le classi secondo CLASS_MAP e restituisce una lista 
    di stringhe formattate per YOLO. 
    Class Map è il dizionario che mappa le classi originali del dataset jhboyo alle classi desiderate nel formato YOLO.
    Se require_vest è True, restituisce una lista vuota se non 
    c'è almeno un'annotazione di vest."""
    converted = []
    has_vest = False
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        old_class = int(float(parts[0]))
        if old_class not in CLASS_MAP:
            continue
        new_class = CLASS_MAP[old_class]
        if new_class == 1:
            has_vest = True
        converted.append(" ".join([str(new_class), *parts[1:5]]))
    if require_vest and not has_vest:
        return []
    return converted


def convert_split(raw_split: Path, output_dir: Path, split: str, require_vest: bool, limit: int) -> int:
    """funzionche che converte uno split del dataset jhboyo in un formato compatibile con YOLO
    serve per permettere di aggiungere un certo numero di immagini e label da uno split specifico 
    del dataset convertito al nuovo dataset finale, con un 
    prefisso uniforme nei nomi dei file e limitando il numero totale di
    immagini aggiunte per evitare squilibri eccessivi. 
    Se require_vest è True, mantiene solo le immagini che 
    contengono almeno un'annotazione di vest."""
    images_dir = raw_split / "images" if (raw_split / "images").exists() else raw_split
    labels_dir = raw_split / "labels" if (raw_split / "labels").exists() else raw_split
    out_images = output_dir / "images" / split
    out_labels = output_dir / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    csv_path = raw_split / "_annotations.csv"
    if csv_path.exists():
        annotations_by_image: dict[str, list[str]] = {}
        has_vest_by_image: dict[str, bool] = {}
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_name = row.get("class", "").strip().lower()
                if class_name not in CSV_CLASS_MAP:
                    continue
                new_class = CSV_CLASS_MAP[class_name]
                filename = row["filename"]
                width = float(row["width"])
                height = float(row["height"])
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_w = (xmax - xmin) / width
                box_h = (ymax - ymin) / height
                annotations_by_image.setdefault(filename, []).append(
                    f"{new_class} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                )
                has_vest_by_image[filename] = has_vest_by_image.get(filename, False) or new_class == 1

        for filename, converted in sorted(annotations_by_image.items()): # ordino per nome del file per avere un ordine coerente
            if limit and count >= limit:
                break
            if require_vest and not has_vest_by_image.get(filename, False):
                continue
            image_path = images_dir / filename
            if not image_path.exists():
                continue
            out_name = f"jhboyo_{split}_{count:05d}_{image_path.name}"
            shutil.copy2(image_path, out_images / out_name) # copio l'immagine con il nuovo nome
            (out_labels / f"{Path(out_name).stem}.txt").write_text("\n".join(converted) + "\n")
            count += 1
        return count

    for label_path in sorted(labels_dir.glob("*.txt")): # ordino per nome del file di label per avere un ordine coerente
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        converted = convert_label(label_path, require_vest=require_vest) # converto le annotazioni del file di label
        if not converted:
            continue

        out_name = f"jhboyo_{split}_{count:05d}_{image_path.name}"
        shutil.copy2(image_path, out_images / out_name)
        (out_labels / f"{Path(out_name).stem}.txt").write_text("\n".join(converted) + "\n") # scrivo il nuovo file di label con le annotazioni convertite
        count += 1
        if limit and count >= limit:
            break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert jhboyo/ppe-dataset to this project's helmet/vest YOLO format.")
    parser.add_argument("--raw-dir", default="external_datasets/ppe_jhboyo_raw")
    parser.add_argument("--output-dir", default="external_datasets/ppe_jhboyo_converted")
    parser.add_argument("--require-vest", action="store_true", help="Keep only images containing at least one vest annotation.")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    counts = {}
    for split, limit in (("train", args.train_limit), ("val", args.val_limit), ("test", args.test_limit)):
        raw_split = split_source(raw_dir, split)
        if raw_split is None:
            print(f"Skipping missing split: {split}")
            continue
        counts[split] = convert_split(
            raw_split=raw_split,
            output_dir=output_dir,
            split=split,
            require_vest=args.require_vest,
            limit=limit,
        )

    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: helmet\n"
        "  1: vest\n"
    )
    (output_dir / "README.md").write_text(
        "# Converted jhboyo PPE Dataset\n\n"
        "Converted from jhboyo/ppe-dataset.\n\n"
        "Class remap:\n\n"
        "- 0 helmet -> 0 helmet\n"
        "- 1 head -> dropped\n"
        "- 2 vest -> 1 vest\n\n"
        f"Require vest: `{args.require_vest}`\n"
        f"Counts: `{counts}`\n"
    )

    print(f"Converted dataset: {output_dir}")
    print(f"Counts: {counts}")
    print(f"Data YAML: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
