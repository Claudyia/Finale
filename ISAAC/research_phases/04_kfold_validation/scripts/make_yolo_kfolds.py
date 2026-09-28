#codice per creare i k-fold stratificati per YOLO, basati sulle classi presenti nelle label.
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
CLASS_NAMES = {0: "helmet", 1: "vest"}


def iter_images(dataset_root: Path) -> list[Path]:
    """""Funzione che itera su tutte le immagini in un dataset YOLO, 
    cercando in images/train e images/val, e restituendo una lista ordinata di percorsi. 
    Supporta estensioni comuni come .jpg, .jpeg e .png."""  
    images = []
    for split in ("train", "val"):
        split_dir = dataset_root / "images" / split
        if not split_dir.exists():
            continue
        for ext in IMAGE_EXTS:
            images.extend(split_dir.glob(f"*{ext}"))
    return sorted(images)


def label_path_for(dataset_root: Path, image_path: Path) -> Path:
    """""Funzione che dato un percorso di immagine, 
    restituisce il percorso del file di label corrispondente,"""
    split = image_path.parent.name
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def read_classes(label_path: Path) -> set[int]:
    """""Funzione che legge un file di label YOLO e restituisce un set di class_id presenti in quel file.
    serve per k-fold stratificato basato sulle classi presenti nelle label."""
    if not label_path.exists():
        return set()
    class_ids: set[int] = set()
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_ids.add(int(float(parts[0])))
        except ValueError:
            continue
    return class_ids


def stratification_key(classes: set[int]) -> str:
    """"Funzione che crea una chiave di stratificazione basata sulle classi presenti in un'immagine.
    chiave è una stringa che concatena i nomi delle classi presenti, ordinati per class_id, separati da +.
    se non ci sono classi, restituisce "negative".
    es. se ci sono classi 0 e 1, chiave sarà "helmet+vest". se c'è solo classe 0, chiave sarà "helmet"."""
    if not classes:
        return "negative"
    names = [CLASS_NAMES.get(class_id, f"class_{class_id}") for class_id in sorted(classes)]
    return "+".join(names)


def build_folds(images: list[Path], dataset_root: Path, k: int, seed: int) -> list[list[Path]]:
    """"Funzione che costruisce i k-fold stratificati basati sulle classi presenti nelle label.
    vuol dire che ogni fold avrà una distribuzione simile di immagini con classi diverse,
    evitando che un fold abbia solo immagini di una classe e un altro fold abbia solo 
    immagini din'altra classe."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        classes = read_classes(label_path_for(dataset_root, image_path))
        grouped[stratification_key(classes)].append(image_path)

    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for _, group_images in sorted(grouped.items()):
        rng.shuffle(group_images)
        for index, image_path in enumerate(group_images):
            folds[index % k].append(image_path)

    for fold in folds:
        fold.sort()
    return folds

"""write_list è una funzione che scrive una lista di percorsi di 
immagini in un file di testo, uno per riga.

write_data_yaml è una funzione che scrive un file data.yaml per YOLO,
specificando il percorso del fold, i file train.txt e val.txt, e i nomi delle classi."""

def write_list(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(str(image.resolve()) for image in images) + "\n")


def write_data_yaml(path: Path, fold_dir: Path) -> None:
    path.write_text(
        f"path: {fold_dir.resolve()}\n"
        "train: train.txt\n"
        "val: val.txt\n\n"
        "names:\n"
        "  0: helmet\n"
        "  1: vest\n"
    )



def main() -> None:
    """Script principale che esegue la creazione dei k-fold stratificati per YOLO.
    Prende in input il percorso del dataset YOLO, il numero di fold, e un seed per la randomizzazione. 
    Verifica che il dataset esista e che l'output directory sia vuota. 
    Poi itera sulle immagini, costruisce i fold stratificati, e scrive i file train.txt,
    val.txt e data.yaml per ogni fold. Infine stampa un riassunto dei fold creati.
    parser è configurato per accettare i parametri da linea di comando, con valori di default.
    
    """
    parser = argparse.ArgumentParser(description="Crea un set di fold k-fold stratificati per YOLO.")
    parser.add_argument(
        "--dataset",
        default="scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost",
        help="YOLO dataset with images/train, images/val, labels/train and labels/val.",
    )
    parser.add_argument(
        "--output-dir",
        default="research_phases/04_kfold_validation/folds/finetune_dataset_ood_hardneg_vestboost_k5",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_dir = Path(args.output_dir)
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    images = iter_images(dataset_root)
    if not images:
        raise FileNotFoundError(f"No images found in dataset: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    folds = build_folds(images, dataset_root, args.folds, args.seed)

    all_images = set(images)
    for index, val_images in enumerate(folds):
        fold_dir = output_dir / f"fold_{index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        val_set = set(val_images)
        train_images = sorted(all_images - val_set)
        write_list(fold_dir / "train.txt", train_images)
        write_list(fold_dir / "val.txt", val_images)
        write_data_yaml(fold_dir / "data.yaml", fold_dir)

    


if __name__ == "__main__":
    main()
