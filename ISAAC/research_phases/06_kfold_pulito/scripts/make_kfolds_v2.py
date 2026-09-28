# Step C (parte 2).
#
# Crea k fold dal merged_manifest.csv usando GroupKFold: tutte le immagini con
# lo stesso group_id (varianti di augmentation, crop, frame dello stesso video)
# finiscono nello stesso fold. In piu i fold sono bilanciati per profilo di
# classe (helmet / vest / helmet+vest / negative).
#
# Non copia immagini: per ogni fold scrive train.txt, val.txt (liste di percorsi)
# e data.yaml. Nessun training.
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def class_profile(n_helmet: int, n_vest: int, is_empty: int) -> str:
    if is_empty or (n_helmet == 0 and n_vest == 0):
        return "negative"
    if n_helmet > 0 and n_vest > 0:
        return "helmet+vest"
    if n_helmet > 0:
        return "helmet"
    return "vest"


ANCHORS = ("dataset_clean/", "external_staging/", "dataset_test_v2/")


def relocate(image_path: str, dataset_root: Path) -> Path:
    """Rimappa il percorso del manifest sotto --dataset-root (per Colab)."""
    posix = Path(image_path).as_posix()
    for anchor in ANCHORS:
        idx = posix.find(anchor)
        if idx != -1:
            return (dataset_root / posix[idx:]).resolve()
    return (dataset_root / Path(image_path).name).resolve()


def build_image_profiles(rows: list[dict], dataset_root: Path) -> dict[Path, str]:
    """Mappa ogni immagine (path relocato) al suo class_profile individuale."""
    profiles: dict[Path, str] = {}
    for row in rows:
        path = relocate(row["image_path"], dataset_root)
        profiles[path] = class_profile(int(row["n_helmet"]), int(row["n_vest"]), int(row["is_empty_label"]))
    return profiles


def oversample_vest(images: list[Path], profiles: dict[Path, str], factor: int) -> list[Path]:
    """Ripete nel training set le immagini vest-only (n_helmet=0, n_vest>0), `factor` volte totali.

    In fase 06 le sorgenti esterne aggiunte (ppe_combined, hardhat_kaggle) sono
    fortemente helmet-heavy e diluiscono vest_ext (il vero vest-boost di fase 03)
    rispetto al dataset piccolo dove funzionava bene: rapporto helmet/vest passato
    da 0.28 a 1.98 (vedi README fase 06). Si ripete solo il path nella lista di
    training, senza toccare il manifest ne duplicare file: nessun rischio leakage.
    """
    if factor <= 1:
        return images
    out: list[Path] = []
    for img in images:
        out.append(img)
        if profiles.get(img) == "vest":
            out.extend([img] * (factor - 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Step C.2: GroupKFold bilanciato dal merged manifest.")
    parser.add_argument("--manifest", default="research_phases/06_kfold_pulito/kfold/merged_manifest.csv")
    parser.add_argument("--out-dir", default="research_phases/06_kfold_pulito/kfold/folds_k5")
    parser.add_argument(
        "--dataset-root",
        default="research_phases/06_kfold_pulito",
        help="Cartella che contiene dataset_clean/, external_staging/, ... "
        "Su Colab: la cartella dove hai scompattato lo zip.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vest-oversample",
        type=int,
        default=3,
        help="Ripete le immagini vest-only (no helmet) N volte nel train.txt di ogni fold, "
        "per contrastare la diluizione da ppe_combined/hardhat_kaggle. 1 = disattivato.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    if args.folds < 2:
        raise ValueError("--folds deve essere >= 2")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}. Esegui prima merge_kfold_manifest.py")

    # Rigenerabile: le liste dipendono solo dal manifest + seed + dataset-root.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(manifest_path.open(newline="")))

    # Raggruppa le righe per group_id.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    # Profilo di un gruppo: prende il profilo "piu informativo" tra le sue immagini.
    priority = {"helmet+vest": 0, "helmet": 1, "vest": 2, "negative": 3}

    def group_profile(members: list[dict]) -> str:
        profiles = {
            class_profile(int(m["n_helmet"]), int(m["n_vest"]), int(m["is_empty_label"]))
            for m in members
        }
        return min(profiles, key=lambda p: priority[p])

    by_profile: dict[str, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        by_profile[group_profile(members)].append(group_id)

    # Assegna i gruppi ai fold: round-robin dentro ogni profilo -> fold bilanciati.
    rng = random.Random(args.seed)
    fold_of_group: dict[str, int] = {}
    for profile, group_ids in sorted(by_profile.items()):
        rng.shuffle(group_ids)
        for index, group_id in enumerate(group_ids):
            fold_of_group[group_id] = index % args.folds

    # Costruisce le liste per ogni fold.
    fold_val_rows: list[list[dict]] = [[] for _ in range(args.folds)]
    for group_id, members in groups.items():
        fold_val_rows[fold_of_group[group_id]].extend(members)

    # Percorsi immagine unici, con il fold assegnato (prima assegnazione vince,
    # cosi un file referenziato da piu righe non finisce in due fold).
    image_to_fold: dict[Path, int] = {}
    for fold_index in range(args.folds):
        for row in fold_val_rows[fold_index]:
            image_to_fold.setdefault(relocate(row["image_path"], dataset_root), fold_index)
    all_images = sorted(image_to_fold)

    missing = [p for p in list(all_images)[:50] if not p.exists()]
    if missing:
        print(f"ATTENZIONE: {len(missing)}/50 immagini campione non trovate sotto {dataset_root}")
        print(f"  es: {missing[0]}")

    profiles = build_image_profiles(rows, dataset_root)
    n_vest_only = sum(1 for p in profiles.values() if p == "vest")

    summary_lines = [
        f"Manifest: {manifest_path}", f"Immagini: {len(rows)}  Gruppi: {len(groups)}",
        f"Vest-only (oversample x{args.vest_oversample}): {n_vest_only}", "",
    ]
    for fold_index in range(args.folds):
        fold_dir = out_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_images = [img for img in all_images if image_to_fold[img] == fold_index]
        val_set = set(val_images)
        train_images = [img for img in all_images if img not in val_set]
        train_images_boosted = oversample_vest(train_images, profiles, args.vest_oversample)

        (fold_dir / "train.txt").write_text("\n".join(str(p) for p in train_images_boosted) + "\n")
        (fold_dir / "val.txt").write_text("\n".join(str(p) for p in val_images) + "\n")
        (fold_dir / "data.yaml").write_text(
            f"path: {fold_dir.resolve()}\n"
            "train: train.txt\n"
            "val: val.txt\n\n"
            "names:\n"
            "  0: helmet\n"
            "  1: vest\n"
        )

        prof = defaultdict(int)
        for row in fold_val_rows[fold_index]:
            prof[class_profile(int(row["n_helmet"]), int(row["n_vest"]), int(row["is_empty_label"]))] += 1
        summary_lines.append(
            f"fold_{fold_index}: train={len(train_images)} (+boost={len(train_images_boosted)}) "
            f"val={len(val_images)}  "
            f"val[helmet={prof['helmet']} vest={prof['vest']} "
            f"helmet+vest={prof['helmet+vest']} neg={prof['negative']}]"
        )

    summary = "\n".join(summary_lines)
    (out_dir / "folds_summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
