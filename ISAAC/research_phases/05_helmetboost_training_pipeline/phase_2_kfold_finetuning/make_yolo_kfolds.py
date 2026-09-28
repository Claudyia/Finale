from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
CLASS_NAMES = {0: "helmet", 1: "vest"}


def iter_images(dataset_root: Path) -> list[Path]:
    images: list[Path] = []
    for split in ("train", "val"):
        split_dir = dataset_root / "images" / split
        if not split_dir.exists():
            continue
        for ext in IMAGE_EXTS:
            images.extend(split_dir.glob(f"*{ext}"))
    return sorted(images)


def label_path_for(dataset_root: Path, image_path: Path) -> Path:
    split = image_path.parent.name
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def read_classes(label_path: Path) -> set[int]:
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
    if not classes:
        return "negative"
    names = [CLASS_NAMES.get(class_id, f"class_{class_id}") for class_id in sorted(classes)]
    return "+".join(names)


def build_folds(images: list[Path], dataset_root: Path, k: int, seed: int) -> list[list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        grouped[stratification_key(read_classes(label_path_for(dataset_root, image_path)))].append(image_path)

    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for _, group_images in sorted(grouped.items()):
        rng.shuffle(group_images)
        for index, image_path in enumerate(group_images):
            folds[index % k].append(image_path)

    for fold in folds:
        fold.sort()
    return folds


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
    parser = argparse.ArgumentParser(description="Phase 2: create stratified YOLO k-fold files.")
    parser.add_argument("--dataset", default="scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost_helmetboost")
    parser.add_argument("--output-dir", default="research_phases/05_helmetboost_training_pipeline/phase_2_kfold_finetuning/folds/helmetboost_k5")
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
        print(f"fold_{index}: train={len(train_images)} val={len(val_images)}")


if __name__ == "__main__":
    main()
