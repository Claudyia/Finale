#codice per creare un nuovo dataset che unisce il dataset di hard-negative
# OOD con il dataset convertito di vest-boost, limitando il numero di 
# immagini aggiunte per evitare squilibri eccessivi. 
# Il nuovo dataset avrà una struttura compatibile con YOLOv8 e includerà un f
# ile data.yaml aggiornato con le nuove classi e i nuovi split.
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def copy_tree_files(src: Path, dst: Path) -> int: 
    """"funzionche che copia tutti i file da una directory di origine a una di destinazione, 
    mantenendo la struttura delle sottocartelle. Restituisce il numero di
    file copiati."""
    count = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        count += 1
    return count


def find_image(images_dir: Path, stem: str) -> Path | None:
    """"funzionche che cerca un'immagine in una directory specificata
    esempio se stem è "img_001" cerca img_001.jpg, img_001.png, ecc. 
    Restituisce il percorso dell'immagine trovata o None se non esiste."""
    
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def add_converted_split(converted: Path, output: Path, split: str, prefix: str, limit: int) -> int:
    """"funzionche che aggiunge un certo numero di
    immagini e label da uno split specifico del dataset convertito
    precisa"""
    images_dir = converted / "images" / split
    labels_dir = converted / "labels" / split
    out_images = output / "images" / split
    out_labels = output / "labels" / split
    count = 0

    if not images_dir.exists() or not labels_dir.exists():
        return 0

    for label_path in sorted(labels_dir.glob("*.txt")):
        if limit and count >= limit:
            break
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        out_name = f"{prefix}_{split}_{count:05d}_{image_path.name}"
        shutil.copy2(image_path, out_images / out_name)
        shutil.copy2(label_path, out_labels / f"{Path(out_name).stem}.txt")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge hard-negative dataset with converted vest-boost dataset.")
    parser.add_argument("--base-dataset", default="scripts/finetunig/dataset/finetune_dataset_ood_hardneg")
    parser.add_argument("--vest-dataset", default="external_datasets/ppe_jhboyo_converted")
    parser.add_argument("--output-dataset", default="scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=400)
    args = parser.parse_args()

    base = Path(args.base_dataset)
    vest = Path(args.vest_dataset)
    output = Path(args.output_dataset)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output dataset is not empty: {output}")

    for split in ("train", "val"):
        copy_tree_files(base / "images" / split, output / "images" / split)
        copy_tree_files(base / "labels" / split, output / "labels" / split)

    added_train = add_converted_split(vest, output, "train", "vestboost", args.train_limit)
    added_val = add_converted_split(vest, output, "val", "vestboost", args.val_limit)

    (output / "data.yaml").write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: helmet\n"
        "  1: vest\n"
    )
    (output / "README.md").write_text(
        "# OOD Hard-Negative + Vest Boost Dataset\n\n"
        f"Base dataset: `{base}`\n"
        f"Vest boost dataset: `{vest}`\n"
        f"Added vest train images: `{added_train}`\n"
        f"Added vest val images: `{added_val}`\n"
    )

    print(f"Created dataset: {output}")
    print(f"Added vest train images: {added_train}")
    print(f"Added vest val images: {added_val}")
    print(f"Data YAML: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
